import cv2
import numpy as np
import os

def calibrate_camera(output_file='data/camera_calibration.yml', grid_size=(7, 7), square_size=25.0):
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

    # Set resolution
    width = 1920  # Desired width (e.g., Full HD)
    height = 1080  # Desired height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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
            try:
                # Open file storage for writing
                fs = cv2.FileStorage(output_file, cv2.FILE_STORAGE_WRITE)
                if not fs.isOpened():
                    raise IOError(f"Failed to open file: {output_file}")

                # Write camera matrix and distortion coefficients
                fs.write("camera_matrix", mtx)
                fs.write("dist_coeffs", dist)
                fs.release()

                print(f"Calibration saved to {output_file}")
            except Exception as e:
                print(f"Failed to write calibration data: {e}")
        else:
            print("Calibration failed.")
    else:
        print("No frames captured for calibration.")

if __name__ == '__main__':
    calibrate_camera()