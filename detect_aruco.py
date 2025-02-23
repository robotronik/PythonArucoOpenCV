# detect_aruco.py
import cv2
import numpy as np
import sys
import os
import threading
import argparse
import time
from flask import Flask, jsonify
from scipy.spatial.transform import Rotation as R

position_data = {"x": None, "y" : None, "theta": None}
status = False # True = running, False = stopped
sucessFrames = 0
failedFrames = 0
app = Flask(__name__)

@app.route('/position', methods=['GET'])
def get_position():
    global status, sucessFrames, failedFrames, position_data
    if (status):
        return jsonify({"position":position_data, "sucessFrames": sucessFrames, "totalFrames": (sucessFrames+failedFrames), "failedFrames": failedFrames})
    else:
        return jsonify({"message": "Camera is not running"})
@app.route('/start', methods=['GET'])
def api_start():
    global status, sucessFrames, failedFrames, position_data
    status = True
    sucessFrames = 0
    failedFrames = 0
    position_data = {"x": None, "y" : None, "theta": None}
    return jsonify({"message": "Starting Camera"})
@app.route('/stop', methods=['GET'])
def api_stop():
    global status
    status = False
    return jsonify({"message": "Stopped Camera"})

def detect_aruco(calib_file, marker_info, cam=0, headless=False):
    """
    Detects ArUco markers and calculates the camera's position relative to the center.
    :param calib_file: Path to the camera calibration file.
    :param marker_info: Dictionary with marker IDs, sizes, and positions.
    """
    if not os.path.exists(calib_file):
        print(f"Calibration file {calib_file} not found.")
        sys.exit(1)

    fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
    mtx = fs.getNode('camera_matrix').mat()
    dist = fs.getNode('dist_coeffs').mat()
    fs.release()
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    
    # General detection parameters
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 23
    aruco_params.adaptiveThreshWinSizeStep = 10
    aruco_params.adaptiveThreshConstant = 7

    # Marker perimeter and corner parameters
    aruco_params.minMarkerPerimeterRate = 0.03
    aruco_params.maxMarkerPerimeterRate = 4.0
    aruco_params.polygonalApproxAccuracyRate = 0.03
    aruco_params.minCornerDistanceRate = 0.05
    aruco_params.minDistanceToBorder = 3
    aruco_params.minMarkerDistanceRate = 0.05

    # Corner refinement parameters (enable subpixel refinement)
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    aruco_params.cornerRefinementWinSize = 5
    aruco_params.relativeCornerRefinmentWinSize = 0.01  # Relative refinement window size
    aruco_params.cornerRefinementMaxIterations = 30
    aruco_params.cornerRefinementMinAccuracy = 0.1

    # Marker border and perspective parameters
    aruco_params.markerBorderBits = 1
    aruco_params.perspectiveRemovePixelPerCell = 4
    aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
    aruco_params.maxErroneousBitsInBorderRate = 0.35
    aruco_params.minOtsuStdDev = 5.0
    aruco_params.errorCorrectionRate = 0.6

    # Inverted marker detection and ArUco3 support
    aruco_params.detectInvertedMarker = False
    aruco_params.useAruco3Detection = True

    cap = cv2.VideoCapture(cam)

    # Set resolution
    width = 1920  # Desired width (e.g., Full HD)
    height = 1080  # Desired height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print("Starting ArUco detection. Press 'q' to quit.")

    global status, sucessFrames, failedFrames, position_data
    status = not headless
    done = False
    sucessFrames = 0
    failedFrames = 0

    while True:
        while(status):
            ret, frame = cap.read()

            scan_res = extract_aruco(ret, frame, mtx, dist, aruco_dict, aruco_params, headless)
            if (scan_res):
                sucessFrames += 1
            else:
                failedFrames += 1

            if (not headless):
                cv2.imshow('ArUco Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                status = False
                done = True

        # Break if done
        if (done):
            break

        # Wait for status to change from rest API
        while(not status):
            time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

def extract_aruco(ret, frame, mtx, dist, aruco_dict, aruco_params, headless):
    global position_data
    if not ret:
        print("Failed to capture frame.")
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None:

        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in marker_info:
                size, global_position = marker_info[marker_id]
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], size, mtx, dist)

                if (not headless):
                    cv2.aruco.drawDetectedMarkers(frame, corners)
                    cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 50)

                # Convert rvec to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rvec) # rvec.flatten().tolist()
                tvec = tvec.reshape(3, 1)
                # Compute camera position in marker's coordinate system
                camera_position = -np.dot(rotation_matrix.T, tvec)

                # Flatten to get a simple (x, y, z) array
                camera_position = camera_position.flatten()
                position_data["x"] = round(global_position[0] - camera_position[1])
                position_data["y"] = round(global_position[1] + camera_position[0])

                # Step 2: Convert rotation matrix to Euler angles
                # Specify the order of axes (e.g., "xyz", "zyx", etc.)
                rotation = R.from_matrix(rotation_matrix)
                euler_angles = rotation.as_euler('zyx', degrees=True)  # 'xyz' or your desired convention
                position_data["theta"] = -euler_angles[0] + 180.0
                print(f"Camera: {position_data}")
                return True
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect ArUco markers and run a REST API.")
    
    # Positional argument for calibration file
    parser.add_argument("calibration_file", type=str, help="Path to the calibration file.")
    
    # Optional arguments
    parser.add_argument("--api-port", type=int, default=5000, help="Port for the REST API (default: 5000).")
    parser.add_argument("--cam", type=int, default=0, help="Camera number to use (default: 0).")
    parser.add_argument("--headless", type=bool, default=False, help="If the program should run headless (default: False).")

    args = parser.parse_args()

    marker_info = {
        33: (100.0, (0.0, 0.0)),   # testing
        20: (100.0, (-400.0, -900.0)),   # Top-right
        21: (100.0, (-400.0, 900.0)),  # Top-left
        22: (100.0, (400.0, -900.0)), # Bottom-left
        23: (100.0, (400.0, -900.0))   # Bottom-right
    }

    # Start REST API in a separate thread
    api_thread = threading.Thread(target=app.run, kwargs={
        "port": args.api_port,
        "debug": False
    })
    api_thread.daemon = True
    api_thread.start()

    detect_aruco(args.calibration_file, marker_info, args.cam, args.headless)