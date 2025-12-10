# Author: Yifei Shao
# Date: 2025/12/05
# Version: 1.0
# Description: Receive face ID from UART and display on LED matrix

from microbit import *

# Configure serial communication
# Initialize UART with a baud rate of 115200, transmit on pin1, receive on pin2
uart.init(baudrate=115200, tx=pin1, rx=pin2)

# Define digit images
NUMBERS = {
    1: Image("00900:00900:00900:00900:00900"),
    2: Image("09990:00090:09990:09000:09990"),
    3: Image("09990:00090:09990:00090:09990"),
    4: Image("09090:09090:09990:00090:00090"),
    5: Image("09990:09000:09990:00090:09990"),
    6: Image("09990:09000:09990:09090:09990"),
    7: Image("09990:00090:00090:00090:00090"),
    8: Image("09990:09090:09990:09090:09990"),
    9: Image("09990:09090:09990:00090:09990")
}

display.show(Image.HAPPY)
sleep(1000)
display.clear()

# Data buffer
buffer = ""

while True:
    # 1. Read serial data
    if uart.any():
        data = uart.read()
        if data:
            try:
                text = str(data, 'UTF-8')
                buffer += text
            except (UnicodeError, ValueError):
                pass

    # 2. Parse the protocol
    if buffer.endswith("#"):
        start_index = buffer.find("$")
        if start_index != -1:
            command = buffer[start_index:]

            try:
                # Logic: Check if recognition was successful
                if "Y" in command:
                    y_index = command.find("Y")
                    # Get the two characters after "Y" as the K210's 0-based ID
                    face_id_str = command[y_index+1 : y_index+3]
                    face_id_0_based = int(face_id_str)

                    # Key correction: Convert 0-based to 1-based display ID
                    display_id = face_id_0_based + 1

                    # Display the corresponding digit on the matrix
                    if display_id in NUMBERS:
                        display.show(NUMBERS[display_id])
                    else:
                        display.show(str(display_id))

                elif "N" in command:
                    # Recognition failed or unknown person
                    display.show(Image.NO)
                    sleep(500)
                    display.clear()

                elif "R" in command:
                    # Registration successful signal
                    display.show(Image.YES)
                    sleep(500)
                    display.clear()

            except Exception:
                display.show("?")

            # Clear the buffer, ready for the next reception
            buffer = ""
        else:
            # Buffer is too long and start was not found, clear to prevent overflow
            if len(buffer) > 50:
                buffer = ""

    sleep(10)
