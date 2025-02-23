# PythonArucoOpenCV

This project detects ArUco markers using OpenCV's ArUco module.

## Installation

Install required libraries using:

```bash
sudo pip install --break-system-packages -r requirements.txt
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

## Notes

- Press 'q' to quit any script.
- Update marker information in `detect_aruco.py` as needed.