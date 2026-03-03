import numpy as np
import cv2 as cv
import os

from background_model import get_background_model, find_best_thresholds, segment_frame_hsv, hsv_background_subtraction

def get_camera_params(cam_id):
    config_file = os.path.join(f"data/cam{cam_id}/intrinsics.xml")
    fs = cv.FileStorage(config_file, cv.FILE_STORAGE_READ)
    if not fs.isOpened():
        print(f"Failed to open {config_file} for {cam_id}.")
        return None
    mtx = fs.getNode("CameraMatrix").mat()
    dist = fs.getNode("DistortionCoeffs").mat()
    rvec = fs.getNode("rvec").mat() if not fs.getNode("rvec").empty() else None
    tvec = fs.getNode("tvec").mat() if not fs.getNode("tvec").empty() else None
    fs.release()
    if rvec is None or tvec is None:
        print(f"Extrinsics not found for {cam_id}. Please ensure XML contains rvec and tvec.")
        return None
    return mtx, dist, rvec, tvec

def get_foreground_mask(cam_id):
    print(f"Processing camera {cam_id}...")
    video_path = os.path.join(f"data/cam{cam_id}/video.avi")
    video_bg_path = os.path.join(f"data/cam{cam_id}/background.avi")
    
    background_model = get_background_model(video_bg_path)
    
    frame = cv.imread(f"data/cam{cam_id}/img_original.png")
    manual = cv.imread(f"data/cam{cam_id}/img_manual.png", cv.IMREAD_GRAYSCALE)

    H, W = background_model.shape[:2]
    frame = cv.resize(frame, (W, H), interpolation=cv.INTER_LINEAR)
    manual = cv.resize(manual, (W, H), interpolation=cv.INTER_NEAREST)

    # Find the best thresholds using grid search
    best_ths, best_err = find_best_thresholds(frame, background_model, manual)
    th_h, th_s, th_v = best_ths
    print(f"Best thresholds: H={th_h}, S={th_s}, V={th_v}, with XOR error: {best_err}")
    
    # Perform background subtraction using the HSV color space
    mask = hsv_background_subtraction(video_path, background_model, th_h, th_s, th_v)
    cv.imshow(f"Foreground Mask - {cam_id}", mask)
    cv.waitKey(200)
    return mask

def reconstruct_voxel(masks, calibration_params, voxel_step=8, grid_dims=(128, 64, 128), min_views=3):
    voxel_lut = {cam: {} for cam in calibration_params.keys()}
    x_range = np.arange(0, grid_dims[0], voxel_step)
    y_range = np.arange(0, grid_dims[1], voxel_step)
    z_range = np.arange(0, grid_dims[2], voxel_step)
    
    for cam_id, params in calibration_params.items():
        mtx, dist, rvec, tvec = params
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    point_3d = np.array([[x, y, z]], dtype=np.float32)
                    imgpt, _ = cv.projectPoints(point_3d, rvec, tvec, mtx, dist)
                    imgpt = imgpt.ravel().astype(int)
                    voxel_lut[cam_id][(x, y, z)] = tuple(imgpt)
    
    voxels_on = []
    for x in x_range:
        for y in y_range:
            for z in z_range:
                count = 0
                for cam_id in calibration_params.keys():
                    imgpt = voxel_lut[cam_id][(x, y, z)]
                    mask = masks[cam_id]
                    h, w = mask.shape
                    u, v = imgpt
                    if u < 0 or u >= w or v < 0 or v >= h:
                        continue
                    if mask[v, u] > 0:
                        count += 1
                if count >= min_views:
                    voxels_on.append([x, y, z])
    return voxels_on

def main():
    cam_ids = [1, 2, 3, 4]
    # use lists so index corresponds to camera order (0-based)
    camera_params = {}

    for cam_id in cam_ids:
        params = get_camera_params(cam_id)
        if params is None:
            print(f"Skipping camera {cam_id} due to missing parameters.")
            return
        else:
            print(f"Camera {cam_id} parameters loaded successfully.")
            camera_params[cam_id] = (params[0], params[1], params[2], params[3])

    # Get background models and masks for each camera
    masks = {}
    for cam_id in cam_ids:
        masks[cam_id] = get_foreground_mask(cam_id)

    voxels = reconstruct_voxel(
        masks, camera_params, voxel_step=8, grid_dims=(128, 64, 128), min_views=3
    )
    print(f"Number of voxels reconstructed: {len(voxels)}")
    
if __name__ == "__main__":
    main()
