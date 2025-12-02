# PythonArucoOpenCV

This project detects ArUco markers using OpenCV's ArUco module.

## Installation

Install required libraries using:

```bash
sudo pip install --break-system-packages -r requirements.txt
sudo apt install python3-picamera2
sudo apt install libcamera-apps

sudo nano /boot/firmware/config.txt
camera_auto_detect=0
dtoverlay=ov9281,cam0
```

## Camera Calibration

Run `calibrate_camera.py` to calibrate your camera. Use a chessboard pattern and capture several frames for accurate calibration. The calibration file will be saved as `data/camera_calibration.yml`.

## Marker Detection

Run `detect_aruco.py` with your calibration file:

```bash
python detect_aruco.py data/camera_calibration.yml
```

Markers should be configured in the script as per their size and positions.

## Requirements

- OpenCV (4.x with ArUco support)
- NumPy


## Raspi with OV9281
On Raspberry Pi 5, automatic camera detection must be disabled for the OV9281.
Run:

sudo nano /boot/firmware/config.txt


Add (or edit) the following lines near the top:

camera_auto_detect=0
dtoverlay=ov9281,cam0


If you plugged into the other CSI connector (CAM1), use ,cam1 instead.

Then save and exit (Ctrl+O, Enter, Ctrl+X).

🔁 Step 3 – Reboot and check detection
sudo reboot

## Notes

- Press 'q' to quit any script.
- Update marker information in `detect_aruco.py` as needed.