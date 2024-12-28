import cv2
import numpy as np
import os

def calibrate_camera(output_file='data/camera_calibration.yml', grid_size=(9, 6), square_size=25.0):
    """
    Calibrates the camera using a chessboard pattern.

    :param output_file: Path to save the calibration file.
    :param grid_size: Number of inner corners per a chessboard row and column.
    :param square_size: Size of a square in mm.
    """
    if not os.path.exists('data'):
        os.makedirs('data')

    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    cap = cv2.VideoCapture(0)

    print("Press 'c' to capture a frame for calibration, and 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

        if ret:
            cv2.drawChessboardCorners(frame, grid_size, corners, ret)

        cv2.imshow('Calibration', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            print("Captured frame for calibration.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) > 0:
        print("Calibrating camera...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        if ret:
            cv2.FileStorage(output_file, cv2.FILE_STORAGE_WRITE).write('camera_matrix', mtx).write('dist_coeffs', dist).release()
            print(f"Calibration saved to {output_file}")
        else:
            print("Calibration failed.")
    else:
        print("No frames captured for calibration.")

if __name__ == '__main__':
    calibrate_camera()