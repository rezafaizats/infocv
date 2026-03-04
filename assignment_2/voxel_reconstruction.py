import numpy as np
import cv2 as cv
import os

from background_model import get_background_model, find_best_thresholds, segment_frame_hsv, hsv_background_subtraction
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

thresholds = {
    1: (40, 80, 60),  # cam 1
    2: (40, 60, 70),  # cam 2
    3: (40, 40, 80),  # cam 3
    4: (20, 50, 60),  # cam 4
}

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

def compute_mask(frame_bgr, bg_bgr, cam_id):
    """
    Computes the foreground mask for a given frame and background model using HSV color space.
    """
    th_h, th_s, th_v = thresholds[cam_id]
    return segment_frame_hsv(frame_bgr, bg_bgr, th_h, th_s, th_v)

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


def reconstruct_voxel(masks, camera_params, voxel_size=8, grid_dims=(64, 64, 64), min_views=2):
    voxel_lut = {cam: {} for cam in camera_params.keys()}
    x_range = np.arange(0, grid_dims[0], voxel_size)
    y_range = np.arange(0, grid_dims[1], voxel_size)
    z_range = np.arange(0, grid_dims[2], voxel_size)

    for cam_id, params in camera_params.items():
        mtx, dist, rvec, tvec = params
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    point_3d = np.array([[x, y, z]], dtype=np.float32)
                    print(f"Projecting voxel ({x:.2f}, {y:.2f}, {z:.2f}) for camera {cam_id}...")
                    imgpt, _ = cv.projectPoints(point_3d, rvec, tvec, mtx, dist)
                    imgpt = imgpt.ravel().astype(int)
                    voxel_lut[cam_id][(x, y, z)] = tuple(imgpt)
    
    voxels_on = []
    for x in x_range:
        for y in y_range:
            for z in z_range:
                count = 0
                for cam_id in camera_params.keys():
                    imgpt = voxel_lut[cam_id][(x, y, z)]
                    mask = masks[cam_id]
                    h, w = mask.shape
                    u, v = imgpt
                    if u < 0 or u >= w or v < 0 or v >= h:
                        continue
                    if mask[v, u] > 0:
                        count += 1
                if count >= min_views:
                    print(f"Adding voxel at ({x:.2f}, {y:.2f}, {z:.2f})")
                    voxels_on.append([x, y, z])
    return voxels_on



def visualize_point_cloud(points, show=True, elev=30, azim=-60):
    if points is None or len(points) == 0:
        print("No points to visualize.")
        return
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='black')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    if show:
        plt.show()


def save_ply(filename, points):
    if points is None or len(points) == 0:
        print("No points to save.")
        return
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
    print(f"Saved {len(points)} points to {filename}")



def main():
    cam_ids = [1, 2, 3, 4]
    data_root = "data"
    
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
    
    # Background substracion 
    background_models = {}
    for cam in cam_ids:
        video_bg_path = os.path.join(data_root, f"cam{cam}/background.avi")
        background_models[cam] = get_background_model(video_bg_path)

    # Videos
    masks = {}
    image_shapes = {}
    for cam in cam_ids:
        # Read a frame from the video to compute the foreground mask
        video_path = os.path.join(data_root, f"cam{cam}/video.avi")
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Can't open the video: {video_path}")
        
        # Read the first frame to compute the foreground mask
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Failed to read a frame from {video_path} to compute foreground mask.")

        image_shapes[cam] = frame.shape[:2]  # Store the image shape for this camera
        # Compute the foreground mask for each camera
        masks[cam] = compute_mask(frame, background_models[cam], cam)


    voxels = reconstruct_voxel(
        masks, camera_params, voxel_size=2, grid_dims=(128, 64, 128), min_views=2
    )
    
    print(f"Number of voxels reconstructed: {len(voxels)}")
    if len(voxels) > 0:
        save_ply('reconstructed.ply', voxels)
        # visualize_point_cloud(voxels)
    
if __name__ == "__main__":
    main()
