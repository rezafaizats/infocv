import numpy as np
import cv2 as cv
import glob
import os

from ClickEvent import get_four_clicks
from Interpolation import linear_interpolation

# Chessboard parameters
calib_flags = 0 

square_size = 0.115
pattern_size = (8, 6)

# Criteria for corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 25, 0.001)

# Prepare 3D object points
objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

# Storage for object and image points
objpoints = []  # 3D points
imgpoints = []  # 2D points

def start_calibration():
    data_root = "data"
    cam = 1  # later cam 2,3,4 
    cam_dir = (f"data\cam{cam}\intrinsics_screenshots\*.png")

    print("Pattern_size:", pattern_size, "square_size(m):", square_size)

    images = get_images(cam_dir)
    img = cv.imread(images[0])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]  # (width, height)

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, pattern_size, None)

        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return corners2, True, False

        else:
            # Manual
            print(f"[MANUAL] OpenCV failed to find corners. Please click 4 OUTER corners on {fname}")
            clicks = get_four_clicks(fname)
            corners2 = linear_interpolation(clicks, pattern_size)
        
            dst_points = np.array([
                    [0, 0],  
                    [pattern_size[1] - 1, 0],  
                    [pattern_size[1] - 1, pattern_size[0] - 1],  
                    [0, pattern_size[0] - 1]  
                ], dtype=np.float32) * square_size
            
            H, _ = cv.findHomography(corners2, dst_points)

            x_grid, y_grid = np.meshgrid(range(pattern_size[1]), range(pattern_size[0]))
            grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T.astype(np.float32) * square_size

            projected_corners = cv.perspectiveTransform(grid_points.reshape(1, -1, 2), np.linalg.inv(H))
            projected_corners = projected_corners.reshape(-1, 1, 2)

            objpoints.append(objp)
            imgpoints.append(projected_corners)

            for point in projected_corners:
                cv.circle(img, tuple(point.ravel().astype(int)), 5, (0, 255, 0), -1)

            cv.imshow("img", img)
            cv.waitKey(500)

    cv.destroyAllWindows()

    img = cv.imread(images[0])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]  # (width, height)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    print(f" Final Reprojection Error: {ret}")
    print(f" Camera Matrix:\n{mtx}")
    print(f" Distortion Coefficients:\n{dist}")


def save_intrinsics_xml(path_out, mtx, dist):
    fs = cv.FileStorage(path_out, cv.FILE_STORAGE_WRITE)
    fs.write("CameraMatrix", mtx)
    fs.write("DistortionCoeffs", dist.reshape(-1, 1))
    fs.release()

def get_images(FilePath: str):
    return glob.glob(FilePath)

def main():
    start_calibration()

if __name__ == "__main__":
    main()