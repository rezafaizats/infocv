import glm
import random
import numpy as np
import cv2 as cv
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INFOCV_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))
VOXEL_MODEL_PATH = "D:/Cloud/infocv/assignment_2/reconstructed.ply"
sys.path.insert(0, INFOCV_DIR)

from background_model import get_background_model, segment_frame_hsv

data_root = os.path.normpath(os.path.join(INFOCV_DIR, "data"))
voxel_data = os.path.normpath(os.path.join(INFOCV_DIR, "reconstructed.ply"))

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
    config_file = os.path.join(cam_dir, "intrinsics.xml")
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
    # for x in range(width):
    #     for y in range(height):
    #         for z in range(depth):
    #             if random.randint(0, 1000) < 5:
    #                 data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
    #                 colors.append([x / width, z / depth, y / height])

    datas, colors = load_ply(voxel_data)
    voxel_points, voxel_colors = [], []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                for data in datas:
                    point = data[:3]  # Extract the XYZ coordinates from the loaded data
                    if point[0] == x and point[1] == y and point[2] == z:  # Check if the current voxel position matches the loaded data
                        print(f"Points match at ({x}, {y}, {z})")
                        voxel_points.append([x * block_size - width/2, y * block_size, z * block_size - depth/2])
                        voxel_colors.append([x / width, z / depth, y / height])  # Assign color based on position

    return voxel_points, voxel_colors


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    
    world_w =  128
    world_h = 64
    world_d = 128

    # use lists so index corresponds to camera order (0-based)
    camera_params = {}
    for cam in cams:
        params = get_camera_params(cam)
        if params is None:
            print(f"Skipping camera {cam} due to missing parameters.")
            return
        else:
            print(f"Camera {cam} parameters loaded successfully.")
            camera_params[cam] = (params[0], params[1], params[2], params[3])
    
    camera_positions = {}
    for cam_id, (mtx, dist, rvec, tvec) in camera_params.items():
        cam_pos = get_camera_world_position(rvec, tvec)
        print(f"Camera {cam_id} world position: {cam_pos}")
        camera_positions[cam_id] = cam_pos

    return [[camera_positions[1][0] * world_w, camera_positions[1][1] * world_h, camera_positions[1][2] * world_d],
            [camera_positions[2][0] * world_w, camera_positions[2][1] * world_h, camera_positions[2][2] * world_d],
            [camera_positions[3][0] * world_w, camera_positions[3][1] * world_h, camera_positions[3][2] * world_d],
            [camera_positions[4][0] * world_w, camera_positions[4][1] * world_h, camera_positions[4][2] * world_d]], \
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    # return [[-64 * block_size, 64 * block_size, 63 * block_size],
    #         [63 * block_size, 64 * block_size, 63 * block_size],
    #         [63 * block_size, 64 * block_size, -64 * block_size],
    #         [-64 * block_size, 64 * block_size, -64 * block_size]], \
    #     [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    camera_params = {}
    for cam in cams:
        params = get_camera_params(cam)
        if params is None:
            print(f"Skipping camera {cam} due to missing parameters.")
            return
        else:
            print(f"Camera {cam} parameters loaded successfully.")
            camera_params[cam] = (params[0], params[1], params[2], params[3])
    
    camera_rotations = {}
    for cam_id, (mtx, dist, rvec, tvec) in camera_params.items():
        cam_rot = get_camera_world_rotation(rvec)
        cam_rot_euler = rotation_matrix_to_euler_angles(cam_rot)
        print(f"Camera {cam_id} world rotation: {cam_rot}")
        camera_rotations[cam_id] = cam_rot_euler

    cam_angles = [camera_rotations[1], camera_rotations[2], camera_rotations[3], camera_rotations[4]]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations

def load_ply(file_path):
    """
    Loads a PLY file and returns the vertices and colors.
    Specifically designed for the ASCII format used in your previous save_point_cloud function.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find where the header ends
    header_end_idx = 0
    num_vertices = 0
    has_color = False
    
    for i, line in enumerate(lines):
        if "element vertex" in line:
            num_vertices = int(line.split()[-1])
        if "property uchar red" in line:
            has_color = True
        if "end_header" in line:
            header_end_idx = i + 1
            break

    # Load the data after the header
    data = np.genfromtxt(lines[header_end_idx:], dtype=np.float32)

    points = data[:, :3] # First 3 columns are X, Y, Z
    colors = data[:, 3:] if has_color else None # Remaining are R, G, B
    
    print(f"Loaded {len(points)} vertices from {file_path}. Check data {points[0]}")

    return points, colors

def get_camera_world_position(rvec, tvec):
    """
    Computes the 3D position of the camera in world coordinates.
    """
    # 1. Convert rotation vector to a 3x3 rotation matrix (R)
    R, _ = cv.Rodrigues(rvec)
    
    # 2. Transpose the rotation matrix (R^T)
    # For rotation matrices, the transpose is equal to the inverse
    R_inv = R.T
    
    # 3. Calculate World Position: C = -R^T * tvec
    camera_position = -R_inv @ tvec
    
    # Return as a simple (X, Y, Z) array
    return camera_position.flatten()

def get_camera_world_rotation(rvec):
    """
    Returns the 3x3 rotation matrix of the camera in world coordinates.
    """
    # 1. Convert vector to matrix (World -> Camera)
    R_world_to_cam, _ = cv.Rodrigues(rvec)
    
    # 2. Invert to get (Camera -> World)
    R_cam_to_world = R_world_to_cam.T
    
    return R_cam_to_world

def rotation_matrix_to_euler_angles(R):
    """
    Converts 3x3 rotation matrix to Euler angles (degrees).
    """
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.degrees([x, y, z])