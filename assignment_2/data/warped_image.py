import cv2
from matplotlib import pyplot as plt
import numpy as np

from Clickevent import get_four_clicks
from interpolation import order_points, linear_interpolation

OUTPUT_DIRECTORY = 'data/cam1/temp/warped.png'
OUTPUT_DIRECTORY2 = 'data/cam1/temp/warped_transformed.png'
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
pattern_size = (8, 6)
square_size = 0.115

def get_checkerboard_top_down(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    clicks = get_four_clicks(image_path)
    clicks2 = get_four_clicks(image_path)
    tl, tr, br, bl = order_points(clicks)
    tl2, tr2, br2, bl2 = order_points(clicks2)

    rows,cols,ch = img.shape

    pts1 = np.float32([tl,bl,tr])
    pts2 = np.float32([tl2,bl2,tr2])
    # pts1 = np.float32([[1026,652],[878,705],[1260,715],[1125,790]])
    # pts2 = np.float32([[220,0],[1675,0],[0,1080],[1675,1080]])
    
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows))
    
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()

    cv2.imwrite(OUTPUT_DIRECTORY, dst)
    
    clicks = get_four_clicks(OUTPUT_DIRECTORY)
    clicks2 = get_four_clicks(OUTPUT_DIRECTORY)
    
    tl, tr, br, bl = order_points(clicks)
    tl2, tr2, br2, bl2 = order_points(clicks2)
    
    pts1 = np.float32([tl,bl,tr,br])
    pts2 = np.float32([tl2,bl2,tr2,br2])
    
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst2 = cv2.warpPerspective(dst, M, (1500,1500))
    
    plt.subplot(121),plt.imshow(dst),plt.title('Input')
    plt.subplot(122),plt.imshow(dst2),plt.title('Output')
    plt.show()

    cv2.imwrite(OUTPUT_DIRECTORY2, dst2)

    clicks = get_four_clicks(OUTPUT_DIRECTORY2)
    corners2 = linear_interpolation(clicks, (8, 6))

    # Convert the selected corners to NumPy array
    corners2 = np.array(corners2, dtype=np.float32)

    # Define object points (real-world coordinates in cm)
    dst_points = np.array([
        [0, 0],  
        [pattern_size[1] - 1, 0],  
        [pattern_size[1] - 1, pattern_size[0] - 1],  
        [0, pattern_size[0] - 1]  
    ], dtype=np.float32) * square_size
    H, _ = cv2.findHomography(corners2, dst_points)

    # Generate grid of expected inner corners
    x_grid, y_grid = np.meshgrid(range(pattern_size[1]), range(pattern_size[0]))
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T.astype(np.float32) * square_size

    # Transform these points to image space
    projected_corners = cv2.perspectiveTransform(grid_points.reshape(1, -1, 2), np.linalg.inv(H))
    projected_corners = projected_corners.reshape(-1, 1, 2)

    # Store points
    # objpoints.append(objp)
    # imgpoints.append(projected_corners)

    # Draw selected and calculated points
    for point in projected_corners:
        cv2.circle(img, tuple(point.ravel().astype(int)), 5, (0, 255, 0), -1)

    cv2.imshow("img", img)

    # if ret:
    #     corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # corners2 = linear_interpolation(clicks, (8, 6))

def check_transformed_corners_manually():
    # Prepare 3D object points
    objp = np.zeros((np.prod(pattern_size), 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    # Storage for object and image points
    objpoints = []  # 3D points
    imgpoints = []  # 2D points

    img = cv2.imread(OUTPUT_DIRECTORY2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, pattern_size, corners, ret)
        cv2.imshow("img", img)
        cv2.waitKey(0)
                
    else:
        print(f"Chessboard corners not found, please select the 4 corners of the chessboard starting from top left and ending at bottom left.")

        # Manually assign the corners
        manual_corners = []
        
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('img', select_corners, {'corners': manual_corners})

        while len(manual_corners) < 4:
            cv2.imshow('img', img)
            cv2.waitKey(1)
        
        # Convert the selected corners to NumPy array
        manual_corners = np.array(manual_corners, dtype=np.float32)

        # Define object points (real-world coordinates in cm)
        dst_points = np.array([
            [0, 0],  
            [pattern_size[1] - 1, 0],  
            [pattern_size[1] - 1, pattern_size[0] - 1],  
            [0, pattern_size[0] - 1]  
        ], dtype=np.float32) * square_size
        H, _ = cv2.findHomography(manual_corners, dst_points)

        # Generate grid of expected inner corners
        x_grid, y_grid = np.meshgrid(range(pattern_size[1]), range(pattern_size[0]))
        grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T.astype(np.float32) * square_size

        # Transform these points to image space
        projected_corners = cv2.perspectiveTransform(grid_points.reshape(1, -1, 2), np.linalg.inv(H))
        projected_corners = projected_corners.reshape(-1, 1, 2)

        # Store points
        objpoints.append(objp)
        imgpoints.append(projected_corners)

        # Draw selected and calculated points
        for point in projected_corners:
            cv2.circle(img, tuple(point.ravel().astype(int)), 5, (0, 255, 0), -1)

        cv2.imshow("img", img)
        cv2.waitKey(0)

# Function to manually select the corners
def select_corners(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(param['corners']) < 4:
            param['corners'].append((x, y))
            print(f"Corner {len(param['corners'])} selected at ({x}, {y})")
            if len(param['corners']) == 4:
                print("All 4 corners selected.")

def main():
    # Run the function
    check_transformed_corners_manually()
    # if result is not None:
    #     cv2.imshow('Warped Board', result)
    #     cv2.waitKey(0)


if __name__ == "__main__":
    main()