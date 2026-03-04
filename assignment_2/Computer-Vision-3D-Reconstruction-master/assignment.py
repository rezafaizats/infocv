import glm
import random
import numpy as np
import cv2 as cv
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INFOCV_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, INFOCV_DIR)

from background_model import get_background_model, segment_frame_hsv

data_root = os.path.normpath(os.path.join(INFOCV_DIR, "assignment_2", "data"))

cams = [1, 2, 3, 4]

thresholds = {
    1: (40, 80, 60),  # cam 1
    2: (40, 60, 70),  # cam 2
    3: (40, 40, 80),  # cam 3
    4: (20, 50, 60),  # cam 4
}

block_size = 1.0


def get_camera_params(cam_id):
    
    cam_dir = os.path.join(data_root, f"cam{cam_id}")
    config_file = os.path.join(cam_dir, "config.xml")
    fs = cv.FileStorage(config_file, cv.FILE_STORAGE_READ)
    if not fs.isOpened():
        print(f"Failed to open {config_file} for {cam_id}.")
        return None
    
    mtx = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeffs").mat()
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


def build_lut(calibration_params, image_shapes, voxel_step=8):
    """
    Reconstructs the 3D voxel positions from the foreground masks and camera calibration parameters.
    """

    world_w =  128
    world_h = 64
    world_d = 128
    
    x_range = np.arange(0, world_w, voxel_step)
    y_range = np.arange(0, world_h, voxel_step)
    z_range = np.arange(0, world_d, voxel_step)

    voxels = [] # 
    for x in x_range:
        for y in y_range:
            for z in z_range:
                # Convert voxel grid coordinates to world coordinates
                X = x * block_size - world_w / 2
                Y = y * block_size
                Z = z * block_size - world_d / 2
                voxels.append([X, Y, Z]) 
    
    voxels_3d = np.array(voxels, dtype=np.float32)  # (N,3)

    uv = {} # to store the projected 2D coordinates of the voxels for each camera
    valid = {} # to keep track of which voxels are valid based on the masks
    
    for cam_id, params in calibration_params.items():
        mtx, dist, rvec, tvec = params
        imgpts, _ = cv.projectPoints(voxels_3d, rvec, tvec, mtx, dist)
        imgpts = imgpts.reshape(-1, 2) # (N_voxels, 2)
        
        # Extract u and v coordinates
        u = imgpts[:, 0].astype(int)
        v = imgpts[:, 1].astype(int)
        
        H, W = image_shapes[cam_id] # Get the image dimensions for this camera

        # Store the projected 2D coordinates for this camera
        uv[cam_id] = (u, v)
        # Mark voxels as valid if they are within the image bounds for this camera
        valid[cam_id] = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    
    return voxels_3d, uv, valid


def voxels_on(masks, voxels_3d, uv, valid, min_views=3):
    """
    Determines which voxels are "on" based on the projected 2D coordinates and the foreground masks.
    """
    # Initialize a list to store the indices of "on" voxels
    on_voxels = []
    
    # Iterate over each voxel
    for i in range(len(voxels_3d)):
        # Count how many cameras see this voxel as foreground
        count = 0
        for cam_id in masks.keys():
            if valid[cam_id][i]:
                u, v = uv[cam_id]
                if masks[cam_id][v[i], u[i]] > 0:  # If the pixel is foreground
                    count += 1
        
        # If the voxel is seen as foreground in at least min_views cameras, mark it as "on"
        if count >= min_views:
            on_voxels.append(i)
    
    return on_voxels

def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    
    # Calibration
    calibration_params = {}
    for cam in cams:
        params = get_camera_params(cam)
        calibration_params[cam] = params

        if params is None:
            raise ValueError(f"Camera {cam} parameters invalid.")

    # Background substracion 
    background_models = {}
    for cam in cams:
        video_bg_path = os.path.join(data_root, f"cam{cam}/background.avi")
        background_models[cam] = get_background_model(video_bg_path)

    # Videos
    masks = {}
    image_shapes = {}
    for cam in cams:
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

    # LUT
    voxels_3d, uv, valid =build_lut(calibration_params, image_shapes, voxel_step=8)
    
    data = voxels_on(masks, voxels_3d, uv, valid, min_views=3)

    colors = []
    for i in range(len(data)):
        colors.append([random.random(), random.random(), random.random()])
    
    return data, colors


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    return [[-64 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, -64 * block_size],
            [-64 * block_size, 64 * block_size, -64 * block_size]], \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations
