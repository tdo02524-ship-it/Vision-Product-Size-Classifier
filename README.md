# Vision-Product-Size-Classifier

Sorts products on a conveyor by size. A camera measures each product, and an
ESP32 triggers a servo to sort it into the correct tray.

## Demo
![System](images/system.png)
<!-- add demo video link -->

## How it works
1. Infrared sensor detects a product on the conveyor.
2. Camera captures it; Python (OpenCV) converts pixels to millimeters.
3. Size result is sent to the ESP32 over UART.
4. ESP32 drives the servo to sort the product.

## Hardware & Technologies
- Hardware: ESP32, camera, conveyor motor, servo, infrared sensor
- Software: Python (OpenCV), C, Qt Designer (GUI), VS Code

## Results
- Measurement error: 0.02 – 0.05 mm
- No missed items or incorrect tray rejections in testing

## How to run
1. `pip install opencv-python ...`
2. Flash the firmware to the ESP32.
3. Run `python main.py`

## Author
Do Ngoc Bao Tam
