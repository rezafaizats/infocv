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

cam = 1  # later cam 2,3,4 

SAVE_INTRINSICS_PATH = f"data/"
SAVE_EXTRINSICS_PATH = f"data/"

show_results = True

def start_calibration():
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
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            if show_results:
                cv.drawChessboardCorners(img, pattern_size, corners, ret)
                cv.imshow("img", img)
                cv.waitKey(200)

        else:
            # Manual
            print(f"[MANUAL] OpenCV failed to find corners. Please click 4 OUTER corners on {fname}")
            clicks = get_four_clicks(fname)
            # corners2 = linear_interpolation(clicks, pattern_size)
        
            corners2 = np.array(clicks, dtype=np.float32)

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
    intrinsics_path = os.path.join(SAVE_INTRINSICS_PATH, f"cam{cam}/intrinsics.xml")
    extrinsics_path = os.path.join(SAVE_EXTRINSICS_PATH, f"cam{cam}/config.xml")
    save_intrinsics_xml(intrinsics_path, mtx, dist, rvecs, tvecs)
    save_extrinsics_xml(extrinsics_path, rvecs, tvecs)
    return mtx, dist, rvecs, tvecs

def draw_axis(image, mtx, dist, rvec, tvec, square_size):
    origin = np.array([[0, 0, 0]], dtype=np.float32)
    x_axis = np.array([[3 * square_size, 0, 0]], dtype=np.float32)
    y_axis = np.array([[0, 3 * square_size, 0]], dtype=np.float32)
    z_axis = np.array([[0, 0, -3 * square_size]], dtype=np.float32)

    imgpts_origin, _ = cv.projectPoints(origin, rvec, tvec, mtx, dist)
    imgpts_x, _ = cv.projectPoints(x_axis, rvec, tvec, mtx, dist)
    imgpts_y, _ = cv.projectPoints(y_axis, rvec, tvec, mtx, dist)
    imgpts_z, _ = cv.projectPoints(z_axis, rvec, tvec, mtx, dist)

    origin_pt = tuple(imgpts_origin.ravel().astype(int))
    x_pt = tuple(imgpts_x.ravel().astype(int))
    y_pt = tuple(imgpts_y.ravel().astype(int))
    z_pt = tuple(imgpts_z.ravel().astype(int))

    cv.line(image, origin_pt, x_pt, (0, 0, 255), 3)
    cv.line(image, origin_pt, y_pt, (0, 255, 0), 3)
    cv.line(image, origin_pt, z_pt, (255, 0, 0), 3)
    
    cv.putText(image, "X", x_pt, cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv.putText(image, "Y", y_pt, cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv.putText(image, "Z", z_pt, cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return image

def save_intrinsics_xml(path, mtx, dist, rvecs, tvecs):
    fs = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)
    fs.write("CameraMatrix", mtx)
    fs.write("DistortionCoeffs", dist.reshape(-1, 1))
    fs.write("rvec", rvecs[0].reshape(3, 1))
    fs.write("tvec", tvecs[0].reshape(3, 1))
    fs.release()

def save_extrinsics_xml(path, rvecs, tvecs):
    fs = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)
    fs.write("rvec", rvecs[0].reshape(3, 1))
    fs.write("tvec", tvecs[0].reshape(3, 1))
    fs.release()


def load_intrinsics(file_path):
    fs = cv.FileStorage(file_path, cv.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("CameraMatrix").mat()
    dist_coeffs = fs.getNode("DistortionCoeffs").mat()
    fs.release()
    return camera_matrix, dist_coeffs

def load_extrinsics(file_path):
    fs = cv.FileStorage(file_path, cv.FILE_STORAGE_READ)
    rvec = fs.getNode("rvec").mat()
    tvec = fs.getNode("tvec").mat()
    fs.release()
    return rvec, tvec

def get_images(FilePath: str):
    return glob.glob(FilePath)

def main():
    # mtx, dist, rvecs, tvecs = start_calibration()
    # rvec = rvecs[0].reshape(3, 1)
    # tvec = tvecs[0].reshape(3, 1)
    intrinsics_file = f"data/cam{cam}/intrinsics.xml"
    extrinsics_file = f"data/cam{cam}/config.xml"
    output_axis_draw = f"data/cam{cam}/axis_visualization.png"
    
    mtx, dist = load_intrinsics(intrinsics_file)
    rvec, tvec = load_extrinsics(extrinsics_file)
    print(f"Loaded rvec and tvec:\n{rvec} \n{tvec}")

    # Choose 1 frame from checkerboard.avi 
    extr_video = os.path.join(f"data/cam{cam}", "checkerboard.avi")
    cap = cv.VideoCapture(extr_video)
    n = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.set(cv.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Cannot read frame for extrinsics")
    
    img_axis = draw_axis(frame, mtx, dist, rvec, tvec, square_size)
    cv.imshow("3D Axes Visualization", img_axis)
    cv.waitKey(0)

    cv.destroyAllWindows()
    
    cv.imwrite(output_axis_draw, img_axis)

if __name__ == "__main__":
    main()