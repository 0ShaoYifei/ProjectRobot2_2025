# Author: Yifei Shao
# Date: 2025/12/07
# Version: 1.2 - Functional Hash: KPU feature converted to list for protection.
# Goal: Stable Face Recognition with operational feature protection.

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

# --- Simple "Hash" / Feature Protection Config ---
# Define the offset value (our 'hash key').
FEATURE_OFFSET = 1234.5678 
# Max score reference for manual similarity (used only for display)
MAX_SIMILARITY_REF = 100.0 

# IMPORTANT: New threshold for manual comparison (L1 norm difference)
# This new threshold is based on the manual distance calculation (list_compare)
# You may need to tune this value on your K210 for optimal performance.
MANUAL_THRESHOLD = 5.0 # For simplicity, a small distance means high similarity

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
dst_point = [(int(38.2946 * FACE_PIC_SIZE / 112), int(51.6963 * FACE_PIC_SIZE / 112)),
             (int(73.5318 * FACE_PIC_SIZE / 112), int(51.5014 * FACE_PIC_SIZE / 112)),
             (int(56.0252 * FACE_PIC_SIZE / 112), int(71.7366 * FACE_PIC_SIZE / 112)),
             (int(41.5493 * FACE_PIC_SIZE / 112), int(92.3655 * FACE_PIC_SIZE / 112)),
             (int(70.7299 * FACE_PIC_SIZE / 112), int(92.2041 * FACE_PIC_SIZE / 112))]

# YOLO V2 anchors for face detection
anchor = (0.1075, 0.126875, 0.126875, 0.175, 0.1465625, 0.2246875, 0.1953125, 0.25375, 0.2440625, 0.351875, 0.341875, 0.4721875, 0.5078125, 0.6696875, 0.8984375, 1.099687, 2.129062, 2.425937)

# --- KPU Model Loading ---
kpu = KPU()
kpu.load_kmodel("/sd/KPU/yolo_face_detect/face_detect_320x240.kmodel")
kpu.init_yolo2(anchor, anchor_num=9, img_w=320, img_h=240, net_w=320 , net_h=240 ,layer_w=10 ,layer_h=8, threshold=0.7, nms_value=0.2, classes=1)

ld5_kpu = KPU()
print("ready load landmark model")
ld5_kpu.load_kmodel("/sd/KPU/face_recognization/ld5.kmodel")

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
    if not start_processing:
        start_processing = True
        time.sleep_ms(BOUNCE_PROTECTION)
key_gpio.irq(set_key_state, GPIO.IRQ_RISING, GPIO.WAKEUP_NOT_SUPPORT)

# List to store features of registered faces (Now storing 'hashed' LISTS of floats)
registered_features = []
THRESHOLD = 80.5 # NOTE: This threshold is now irrelevant, MANUAL_THRESHOLD is used.
recog_flag = False

# --- New Function: Manual Similarity Check ---
# Uses Manhattan Distance (L1 norm) to determine similarity between two feature LISTS.
def list_compare_distance(f1_list, f2_list):
    """Calculates Manhattan distance (L1) and returns it as a DIS-similarity score."""
    distance = 0.0
    for a, b in zip(f1_list, f2_list):
        distance += abs(a - b)
    # NOTE: Lower 'distance' means higher similarity.
    return distance

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
    gc.collect() 
    clock.tick()
    img = sensor.snapshot()
    kpu.run_with_output(img)
    dect = kpu.regionlayer_yolo2()
    fps = clock.fps()
    
    if len(dect) > 0:
        for l in dect :
            x1, y1, cut_img_w, cut_img_h= extend_box(l[0], l[1], l[2], l[3], scale=0)
            face_cut = img.cut(x1, y1, cut_img_w, cut_img_h)
            face_cut_128 = face_cut.resize(128, 128)
            face_cut_128.pix_to_ai()

            # 2a. Get facial landmarks (5 points)
            out = ld5_kpu.run_with_output(face_cut_128, getlist=True)
            face_key_point = []
            for j in range(5):
                x = int(KPU.sigmoid(out[2 * j])*cut_img_w + x1)
                y = int(KPU.sigmoid(out[2 * j + 1])*cut_img_h + y1)
                face_key_point.append((x,y))

            # 2b. Align the face
            T = image.get_affine_transform(face_key_point, dst_point)
            image.warp_affine_ai(img, feature_img, T)
            del face_key_point

            # 2c. Extract the original feature vector (KPU object)
            feature_kpu_object = fea_kpu.run_with_output(feature_img, get_feature = True)
            
            # --- FEATURE HASHING AND CONVERSION (The Fix) ---
            # 1. Convert the KPU object to a Python list
            feature_list = list(feature_kpu_object)
            
            # 2. Apply the simple "hash" (arithmetic operation) to the list
            # Note: This is now safe as we are operating on a standard list.
            hashed_feature_list = [f + FEATURE_OFFSET for f in feature_list]
            
            # 3. Release the original KPU object early for memory
            del feature_kpu_object 
            
            # --- Recognition Logic (Manual Comparison) ---
            min_distance = float('inf')
            index = -1
            
            for j in range(len(registered_features)):
                # Compare the current HASHED list with the stored HASHED list
                current_distance = list_compare_distance(registered_features[j], hashed_feature_list)
                
                if current_distance < min_distance:
                    min_distance = current_distance
                    index = j
            
            # Convert distance to a conceptual score for display (higher is better)
            # We must invert the distance to get a "similarity" score
            # The exact max score is complex, so we'll use a fixed value relative to the threshold for display
            conceptual_score = MANUAL_THRESHOLD - (min_distance / 1000.0)
            
            # 2d. Find best match
            if index != -1 and min_distance < MANUAL_THRESHOLD:
                # Recognized! (Distance is less than the manual threshold)
                img.draw_string(0, 195, "Persion ID: %d, Dist: %2.2f" %(index, min_distance), color=(0, 255, 0), scale=2)
                recog_flag = True
            elif index != -1:
                # Detected but Stranger (Distance too high)
                img.draw_string(0, 195, "Unreg, Dist: %2.2f" %(min_distance), color=(255, 0, 0), scale=2)
            
            # --- Registration Logic (Button Check) ---
            if start_processing:
                # Store the HASHED feature list
                registered_features.append(hashed_feature_list) 
                print("Total registered faces: %d" % len(registered_features))
                start_processing = False # Reset flag

            # --- Prepare Serial Message & Draw Box ---
            if recog_flag:
                img.draw_rectangle(l[0],l[1],l[2],l[3], color=(0, 255, 0))
                recog_flag = False
                
                if index < 10:
                    index="%02d"%index
                
                msg_ = "%s%s"%("Y",str(index))
            else:
                img.draw_rectangle(l[0],l[1],l[2],l[3], color=(255, 255, 255))
                msg_ = "N"
                
            del (face_cut_128)
            del (face_cut)
            
    # --- Step 3: Send Serial Data ---
    if len(dect) > 0:
       send_data ="$"+"08"+ msg_+','+"#" 
       time.sleep_ms(5)
       serial.send(send_data)
    else :
        serial.send("#")

    # --- Step 4: Display info on LCD ---
    img.draw_string(0, 0, "%2.1ffps" %(fps), color=(0, 60, 255), scale=2.0)
    img.draw_string(0, 215, "Press BOOT key to register new face", color=(255, 100, 0), scale=2.0)
    lcd.display(img)

# --- Cleanup at end of program (never reached in infinite loop) ---
kpu.deinit()
ld5_kpu.deinit()
fea_kpu.deinit()