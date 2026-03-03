import glm
import random
import numpy as np
import os
import glob
import cv2 as cv

block_size = 4.0
VOXEL_MODEL_PATH = "D:/Cloud/infocv/assignment_2/reconstructed.ply"

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
    # data, colors = [], []
    datas, colors = load_ply(VOXEL_MODEL_PATH)
    voxel_points, voxel_colors = [], []
    # for x in range(width):
    #     for y in range(height):
    #         for z in range(depth):
    #             if random.randint(0, 1000) < 5:
    #                 data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
    #                 colors.append([x / width, z / depth, y / height])
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

def get_camera_params(cam_id):
    config_file = os.path.join(f"D:/Cloud/infocv/assignment_2/data/cam{cam_id}/intrinsics.xml")
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
    