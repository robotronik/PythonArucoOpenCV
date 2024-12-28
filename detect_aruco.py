# detect_aruco.py
import cv2
import numpy as np
import sys
import os
import threading
import time
from flask import Flask, jsonify

position_data = {"position": None, "rotation": None}
app = Flask(__name__)

@app.route('/position', methods=['GET'])
def get_position():
    return jsonify(position_data)

def detect_aruco(calib_file, marker_info):
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

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters_create()

    cap = cv2.VideoCapture(0)

    print("Starting ArUco detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if ids is not None:
            detected_positions = []

            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in marker_info:
                    size, global_position = marker_info[marker_id]
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], size, mtx, dist)

                    cv2.aruco.drawDetectedMarkers(frame, corners)
                    cv2.aruco.drawAxis(frame, mtx, dist, rvec, tvec, 50)

                    marker_camera_position = tvec.flatten()  # Camera position relative to marker

                    # Calculate camera position relative to the center
                    camera_position = [
                        global_position[0] - marker_camera_position[0],
                        global_position[1] - marker_camera_position[1],
                        global_position[2] - marker_camera_position[2]
                    ]
                    detected_positions.append(camera_position)

            if detected_positions:
                # Average position from all detected markers
                avg_position = np.mean(detected_positions, axis=0).tolist()
                position_data["position"] = avg_position
                position_data["rotation"] = rvec.flatten().tolist()

                print(f"Camera Position: {avg_position}, Rotation: {position_data['rotation']}")

        cv2.imshow('ArUco Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python detect_aruco.py <calibration_file>")
        sys.exit(1)

    marker_info = {
        1: (100, (600, 600, 0)),   # Top-right
        2: (100, (-600, 600, 0)),  # Top-left
        3: (100, (-600, -600, 0)), # Bottom-left
        4: (100, (600, -600, 0))   # Bottom-right
    }

    # Start REST API in a separate thread
    api_thread = threading.Thread(target=app.run, kwargs={"port": 5000, "debug": False})
    api_thread.daemon = True
    api_thread.start()

    detect_aruco(sys.argv[1], marker_info)
