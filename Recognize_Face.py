# Author:Yifei Shao
# Date: 2025/12/06
# Version: 1.1 - Minor Refactor and Styling
# Goal: Face Recognition for my K210 Robot Project

# Core imports for sensor, image processing, and KPU
import sensor, image, time, lcd
import gc
from maix import KPU
from maix import GPIO, utils
from fpioa_manager import fm
from board import board_info
# Custom serial module for communication
from modules import ybserial
import time

# Initialize serial communication handler
serial = ybserial()

# --- Camera & Display Initialization ---
lcd.init()
sensor.reset()
sensor.set_framesize(sensor.QVGA) # Set frame size to 320x240
sensor.set_pixformat(sensor.RGB565) # Set pixel format
sensor.skip_frames(time = 100) # Give the camera a moment to settle
clock = time.clock()

# Dedicated image buffer for feature extraction (64x64)
FACE_PIC_SIZE = 64
feature_img = image.Image(size=(FACE_PIC_SIZE, FACE_PIC_SIZE), copy_to_fb=False)
feature_img.pix_to_ai()

# Define destination points for face alignment (normalized to 112x112, then scaled to FACE_PIC_SIZE)
# These points correspond to the 5 facial landmarks (eyes, nose, mouth corners)
dst_point =[(int(38.2946 * FACE_PIC_SIZE / 112), int(51.6963 * FACE_PIC_SIZE / 112)),
            (int(73.5318 * FACE_PIC_SIZE / 112), int(51.5014 * FACE_PIC_SIZE / 112)),
            (int(56.0252 * FACE_PIC_SIZE / 112), int(71.7366 * FACE_PIC_SIZE / 112)),
            (int(41.5493 * FACE_PIC_SIZE / 112), int(92.3655 * FACE_PIC_SIZE / 112)),
            (int(70.7299 * FACE_PIC_SIZE / 112), int(92.2041 * FACE_PIC_SIZE / 112)) ]

# YOLO V2 anchors for face detection
anchor = (0.1075, 0.126875, 0.126875, 0.175, 0.1465625, 0.2246875, 0.1953125, 0.25375, 0.2440625, 0.351875, 0.341875, 0.4721875, 0.5078125, 0.6696875, 0.8984375, 1.099687, 2.129062, 2.425937)

# --- KPU Model Loading ---

# 1. Face Detection Model
kpu = KPU()
kpu.load_kmodel("/sd/KPU/yolo_face_detect/face_detect_320x240.kmodel")
kpu.init_yolo2(anchor, anchor_num=9, img_w=320, img_h=240, net_w=320 , net_h=240 ,layer_w=10 ,layer_h=8, threshold=0.7, nms_value=0.2, classes=1)

# 2. 5-point Landmark Detection Model (ld5)
ld5_kpu = KPU()
print("ready load landmark model")
ld5_kpu.load_kmodel("/sd/KPU/face_recognization/ld5.kmodel")

# 3. Feature Extraction Model
fea_kpu = KPU()
print("ready load feature model")
fea_kpu.load_kmodel("/sd/KPU/face_recognization/feature_extraction.kmodel")

# --- Setup for Face Registration (via Boot Key) ---
start_processing = False
BOUNCE_PROTECTION = 50 # Debounce time in ms

fm.register(board_info.BOOT_KEY, fm.fpioa.GPIOHS0)
key_gpio = GPIO(GPIO.GPIOHS0, GPIO.IN)

# Interrupt handler for key press
def set_key_state(*_):
    global start_processing
    # Only set flag if we haven't just processed a press
    if not start_processing:
        start_processing = True
        time.sleep_ms(BOUNCE_PROTECTION)
key_gpio.irq(set_key_state, GPIO.IRQ_RISING, GPIO.WAKEUP_NOT_SUPPORT)

# List to store features of registered faces
registered_features = []
THRESHOLD = 80.5 # Recognition threshold (similarity score)
recog_flag = False

# Function to safely extend the bounding box
def extend_box(x, y, w, h, scale):
    x1_t = x - scale*w
    x2_t = x + w + scale*w
    y1_t = y - scale*h
    y2_t = y + h + scale*h
    
    # Clip coordinates to be within the image bounds (320x240)
    x1 = int(x1_t) if x1_t>1 else 1
    x2 = int(x2_t) if x2_t<320 else 319
    y1 = int(y1_t) if y1_t>1 else 1
    y2 = int(y2_t) if y2_t<240 else 239
    
    cut_img_w = x2-x1+1
    cut_img_h = y2-y1+1
    return x1, y1, cut_img_w, cut_img_h

msg_=""
while True:
    # Important: Collect garbage to free up memory before processing
    gc.collect() 
    clock.tick()
    img = sensor.snapshot()

    # Step 1: Run Face Detection
    kpu.run_with_output(img)
    dect = kpu.regionlayer_yolo2() # Get detection results (bounding boxes)
    fps = clock.fps()
    
    # Step 2: Process Detected Faces
    if len(dect) > 0:
        for l in dect :
            # l is [x, y, w, h, score]
            # Get extended/clipped bounding box
            x1, y1, cut_img_w, cut_img_h= extend_box(l[0], l[1], l[2], l[3], scale=0)
            face_cut = img.cut(x1, y1, cut_img_w, cut_img_h)
            face_cut_128 = face_cut.resize(128, 128)
            face_cut_128.pix_to_ai()

            # 2a. Get facial landmarks (5 points)
            out = ld5_kpu.run_with_output(face_cut_128, getlist=True)
            face_key_point = []
            for j in range(5):
                # Map normalized landmark coordinates back to the original image coordinates
                x = int(KPU.sigmoid(out[2 * j])*cut_img_w + x1)
                y = int(KPU.sigmoid(out[2 * j + 1])*cut_img_h + y1)
                face_key_point.append((x,y))

            # 2b. Align the face using affine transformation
            T = image.get_affine_transform(face_key_point, dst_point)
            image.warp_affine_ai(img, feature_img, T)
            del face_key_point # Free memory

            # 2c. Extract the feature vector
            feature = fea_kpu.run_with_output(feature_img, get_feature = True)
            
            # --- Recognition Logic ---
            scores = []
            for j in range(len(registered_features)):
                # Compare current feature with all registered features
                score = kpu.feature_compare(registered_features[j], feature)
                scores.append(score)
            
            # 2d. Find best match
            if len(scores):
                max_score = max(scores)
                index = scores.index(max_score) # ID of the recognized face (0-based)
                
                if max_score > THRESHOLD:
                    # Recognized a known face!
                    img.draw_string(0, 195, "Persion ID: %d, Score: %2.1f" %(index, max_score), color=(0, 255, 0), scale=2)
                    recog_flag = True
                else:
                    # Found a face, but it's not recognized (stranger)
                    img.draw_string(0, 195, "Unregistered, Score: %2.1f" %(max_score), color=(255, 0, 0), scale=2)
            del scores # Free memory
            
            # --- Registration Logic (Button Check) ---
            if start_processing:
                registered_features.append(feature)
                print("Total registered faces: %d" % len(registered_features))
                start_processing = False # Reset flag

            # --- Prepare Serial Message & Draw Box ---
            if recog_flag:
                # Green box for recognized person
                img.draw_rectangle(l[0],l[1],l[2],l[3], color=(0, 255, 0))
                recog_flag = False
                
                # Format ID to be two digits (e.g., 00, 01, 10)
                if index < 10:
                    index="%02d"%index
                
                # Recognition successful message: "Y" + 2-digit ID
                msg_ = "%s%s"%("Y",str(index))
            else:
                # White box for detected but unrecognized face
                img.draw_rectangle(l[0],l[1],l[2],l[3], color=(255, 255, 255))
                # Recognition failure/unregistered message
                msg_ = "N"
                
            del (face_cut_128) # Free memory
            del (face_cut)     # Free memory
            
    # --- Step 3: Send Serial Data ---
    if len(dect) > 0:
       # Protocol: $<DeviceID> <Message> , #
       send_data ="$"+"08"+ msg_+','+"#" 
       time.sleep_ms(5)
       serial.send(send_data)
    else :
        # Send a simple separator when no face is detected
        serial.send("#")

    # --- Step 4: Display info on LCD ---
    img.draw_string(0, 0, "%2.1ffps" %(fps), color=(0, 60, 255), scale=2.0)
    img.draw_string(0, 215, "Press BOOT key to register new face", color=(255, 100, 0), scale=2.0)
    lcd.display(img)

# --- Cleanup at end of program (never reached in infinite loop) ---
kpu.deinit()
ld5_kpu.deinit()
fea_kpu.deinit()