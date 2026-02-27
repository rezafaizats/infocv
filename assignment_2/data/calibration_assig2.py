import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob

from Clickevent import get_four_clicks
from interpolation import linear_interpolation

INTRINSIC_CALIBRATION_IMAGES_PATH = 'data\cam1\intrinsics_screenshots\*.png'
EXTRINSIC_CALIBRATION_IMAGES_PATH = 'data\cam1\checkerboard.png'

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

calib_flags = 0 

square_size = 0.115
pattern_size = (8, 6)



# -----------------------------------------------------------
# 1) CORNERS  
def get_corners_from_frame(frame, pattern_size, allow_manual, tmp_path="_temp_click.png"):
    """
    - recibs frame 
    - if it fails, it saves the PNG temporal and uses the clicks
    """
    
    img = cv.imread(frame)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, pattern_size, None)

    if ret:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return corners2, True, False

    if not allow_manual:
        return None, False, False

    # Manual
    # cv.imwrite(tmp_path, frame)
    # print(f"[MANUAL] OpenCV failed. Click 4 OUTER corners on {tmp_path}")
    clicks = get_four_clicks(frame)
    corners2 = linear_interpolation(clicks, pattern_size)
    return corners2, True, True


# -----------------------------------------------------------
# 2) CALIBRATION 
def calibrate_run(corners_list, objp, image_size):
    """
    corners_list: list od corners2 (1 per valid frame)
    """
    objpoints = []
    imgpoints = []

    for corners2 in corners_list:
        objpoints.append(objp)
        imgpoints.append(corners2)

    rms, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=calib_flags
    )
    return rms, mtx, dist, rvecs, tvecs


def reprojection_errors(corners_list, objp, mtx, dist, rvecs, tvecs):
    """
    We index per list
    """
    errors = []
    for i in range(len(corners_list)):
        imgpoints = corners_list[i]
        proj, _ = cv.projectPoints(objp, rvecs[i], tvecs[i], mtx, dist)
        e = cv.norm(imgpoints, proj, cv.NORM_L2) / len(proj)
        errors.append(float(e))
    return errors, float(np.mean(errors))


def iterative_rejection(corners_list, objp, image_size, thr=0.8):
    """
    Take out the frame with worse reprojection error and recalibrate.
    """
    current = corners_list.copy()
    prev_mean = None

    while True:
        rms, mtx, dist, rvecs, tvecs = calibrate_run(current, objp, image_size)
        per_frame, mean_e = reprojection_errors(current, objp, mtx, dist, rvecs, tvecs)

        worst_i = int(np.argmax(per_frame))
        worst_e = per_frame[worst_i]

        if worst_e <= thr:
            break
        if prev_mean is not None and mean_e >= prev_mean:
            break

        current.pop(worst_i)
        prev_mean = mean_e

        if len(current) < 8:  
            break

    return current


# -----------------------------------------------------------
# 3) EXTRACT FRAMES FROM THE VIDEO 
def sample_frames_from_video(video_path, num_samples=30):
    """
    Take num_samples frames from the video
    """
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No video found on {video_path}")

    n = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        raise RuntimeError("Can´t read CAP_PROP_FRAME_COUNT")

    idxs = np.linspace(0, n - 1, num_samples, dtype=int)
    frames = []

    for idx in idxs:
        cap.set(cv.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)

    cap.release()
    return frames

def remove_distortion(img, mtx, dist):
    height, width = img.shape[:2]
    new_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
    img_undist = cv.undistort(img, mtx, dist, None, new_mtx)

    # crop the image
    x, y, w, h = roi
    dst = img_undist[y:y+h, x:x+w]

    cv.line(img, (1769, 103), (1780, 922), (255, 255, 255), 2)
    cv.line(dst, (1769, 103), (1780, 922), (255, 255, 255), 2)

    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(dst)
    plt.show()
    return dst

def draw_axis(img, origin, xpt, ypt, zpt):
    """Draw XYZ axes on the image."""
    o = tuple(origin.ravel().astype(int))
    x = tuple(xpt.ravel().astype(int))
    y = tuple(ypt.ravel().astype(int))
    z = tuple(zpt.ravel().astype(int))

    img = cv.line(img, o, x, (0, 0, 255), 3)   # X red
    img = cv.line(img, o, y, (0, 255, 0), 3)   # Y green
    img = cv.line(img, o, z, (255, 0, 0), 3)   # Z blue
    return img

# -----------------------------------------------------------
# 4) SAVE XML 
def save_intrinsics_xml(path_out, mtx, dist):
    fs = cv.FileStorage(path_out, cv.FILE_STORAGE_WRITE)
    fs.write("CameraMatrix", mtx)
    fs.write("DistortionCoeffs", dist.reshape(-1, 1))
    fs.release()


def save_config_xml(path_out, mtx, dist, rvec, tvec):
    """
    config.xml for voxel reconstruction (tags generics).
    """
    fs = cv.FileStorage(path_out, cv.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", mtx)
    fs.write("dist_coeffs", dist.reshape(-1, 1))
    fs.write("rvec", rvec.reshape(3, 1))
    fs.write("tvec", tvec.reshape(3, 1))
    fs.release()


def main():
    data_root = "data"
    cam = 1  # later cam 2,3,4 
    cam_dir = os.path.join(data_root, f"cam{cam}")

    print("Pattern_size:", pattern_size, "square_size(m):", square_size)

    # Create objp 
    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    # -------------------------
    # A) INTRINSICS
    # -------------------------
    intr_video = os.path.join(cam_dir, "intrinsics.avi")
    # frames = sample_frames_from_video(intr_video, num_samples=30)
    frames = glob.glob(INTRINSIC_CALIBRATION_IMAGES_PATH)

    corners_list = []
    for frame in frames:
        corners2, found, used_manual = get_corners_from_frame(
            frame, pattern_size, allow_manual=False, # don´t take clicks
            # tmp_path=f"_temp_intr_cam{cam}_{i}.png"
        )
        if found and corners2 is not None:
            corners_list.append(corners2)

    if len(corners_list) < 8:
        raise RuntimeError(f"Few frames available for intrinsics: {len(corners_list)}")
    
    img = cv.imread(frames[0])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # h, w = frames[0].shape[:2]
    image_size =  gray.shape[::-1]

    # Calibration
    rms, mtx, dist, rvecs, tvecs = calibrate_run(corners_list, objp, image_size)
    print("\n[INTRINSICS] RMS:", rms)
    print("[INTRINSICS] K:\n", mtx)
    print("[INTRINSICS] dist:", dist.ravel())

    # Clicks
    corners_filtered = iterative_rejection(corners_list, objp, image_size, thr=0.8)
    if len(corners_filtered) != len(corners_list):
        rms, mtx, dist, rvecs, tvecs = calibrate_run(corners_filtered, objp, image_size)
        print("\n[INTRINSICS - filtered] kept:", len(corners_filtered), "RMS:", rms)

    # Save intrinsics.xml
    intr_xml = os.path.join(cam_dir, "intrinsics.xml")
    save_intrinsics_xml(intr_xml, mtx, dist)
    print("[SAVED]", intr_xml)

    
    # -------------------------
    # B) EXTRINSICS
    # -------------------------
    # Choose 1 frame from checkerboard.avi 
    extr_video = os.path.join(cam_dir, "checkerboard.avi")
    cap = cv.VideoCapture(extr_video)
    n = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.set(cv.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Cannot read frame for extrinsics")

    img_undist = remove_distortion(frame, mtx, dist)
    # frame = glob.glob(EXTRINSIC_CALIBRATION_IMAGES_PATH)
    frame = cv.imread(EXTRINSIC_CALIBRATION_IMAGES_PATH)

    # allow_manual=True for clicks, as OpencV can fail
    corners2, found, used_manual = get_corners_from_frame(
        EXTRINSIC_CALIBRATION_IMAGES_PATH, pattern_size, allow_manual=True,
        tmp_path=f"_temp_extr_cam{cam}.png"
    )
    if not found or corners2 is None:
        raise RuntimeError("Couldn´t detect corners for extrinsics")

    ok, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist)
    if not ok:
        raise RuntimeError("solvePnP failed")

    print("\n[EXTRINSICS] rvec:", rvec.ravel())
    print("[EXTRINSICS] tvec:", tvec.ravel())

    # Save config.xml
    cfg_xml = os.path.join(cam_dir, "config.xml")
    save_config_xml(cfg_xml, mtx, dist, rvec, tvec)
    print("[SAVED]", cfg_xml)

    
    # axis
    axis_len = axis_len = 6 * square_size
    axis = np.float32([[0,0,0], [axis_len,0,0], [0,axis_len,0], [0,0,-axis_len]])

    imgpts_axis, _ = cv.projectPoints(axis, rvec, tvec, mtx, dist)
    img = draw_axis(frame, imgpts_axis[0], imgpts_axis[1], imgpts_axis[2], imgpts_axis[3])
    cv.imshow("Axis", img)
    cv.waitKey(0)
    
    cv.destroyAllWindows()



if __name__ == "__main__":
    main()