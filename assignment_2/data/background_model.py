import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import argparse


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
FILE_BACKGROUND_PATH = "assignment_2/data/cam1/background.avi"
FILE_VIDEO_PATH = "assignment_2/data/cam1/video.avi"

def get_background_model(video_path: str, num_frames: int = 100):
    """
    Computes the background model by averaging the first num_frames frames of the video.

    Returns: the background model as a numpy array (image).
    """
    # read the video
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Can't open the video: {video_path}")

    frames = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Only {i} frames were read from the video.")
            break
        frames.append(frame.astype(np.float32))

    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames were read from the video to compute the background model")

    background_model = np.mean(frames, axis=0).astype(np.uint8)
    return background_model


def hsv_background_subtraction(video_path, background_model, th_h=30, th_s=50, th_v=50):
    """
    Performs background subtraction using HSV color space.
    We have 3 thresholds for hue, saturation, and value to determine if a pixel belongs to the foreground.
    We choose them base on the expected variations in the background and the desired sensitivity to changes.
    Returns: a binary mask of the foreground objects.
    """
    # read the video
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Can't open the video: {video_path}")

    # Convert background model to HSV color space
    hsv_background = cv.cvtColor(background_model, cv.COLOR_BGR2HSV)
    # Split the background model into H, S, and V channels
    hb, sb, vb = cv.split(hsv_background) 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # convert the current frame to HSV color space
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv_frame)
        
        # hue difference should consider the circular nature of hue values (0-179 in OpenCV) ??
        diff_h = cv.absdiff(h, hb)
        diff_h = np.minimum(diff_h, 180 - diff_h) # circular hue difference
        diff_s = cv.absdiff(s, sb)
        diff_v = cv.absdiff(v, vb)

        # Convert differences to binary masks(0 or 1) based on thresholds and then to 0 or 255 values
        mask_h = (diff_h > th_h).astype(np.uint8) * 255 
        mask_s = (diff_s > th_s).astype(np.uint8) * 255     
        mask_v = (diff_v > th_v).astype(np.uint8) * 255
        # - 0 → the pixel belongs to the background 
        # - 255 → the pixel belongs to the foreground

        # We combine the masks using a logical OR op to get the final binary foreground mask
        foreground_mask = (mask_h | mask_s | mask_v)

        # POST-PROCESSING: Apply morphological operations to reduce noise and improve the foreground mask
        # We use an elliptical kernel because ??
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

        # We use morphological opening to remove small noise and closing to fill small holes in the detected foreground objects.
        foreground_mask = cv.morphologyEx(foreground_mask, cv.MORPH_OPEN, kernel, iterations=1)
        foreground_mask = cv.morphologyEx(foreground_mask, cv.MORPH_CLOSE, kernel, iterations=2)
        

        # Display results
        cv.imshow('Frame', frame)
        cv.imshow('Foreground Mask', foreground_mask)

        if cv.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def segment_frame_hsv(frame_bgr, background_bgr, th_h, th_s, th_v):
    """
    Your HSV background subtraction for a single frame.
    Returns: binary mask 0/255 (uint8)
    """
    hsv_bg = cv.cvtColor(background_bgr, cv.COLOR_BGR2HSV)
    hb, sb, vb = cv.split(hsv_bg)

    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    # Hue diff (circular)
    diff_h = cv.absdiff(h, hb)
    diff_h = np.minimum(diff_h, 180 - diff_h)

    diff_s = cv.absdiff(s, sb)
    diff_v = cv.absdiff(v, vb)

    mask_h = (diff_h > th_h).astype(np.uint8) * 255
    mask_s = (diff_s > th_s).astype(np.uint8) * 255
    mask_v = (diff_v > th_v).astype(np.uint8) * 255

    # Simple combination (you can change later)
    fg = (mask_h | mask_s | mask_v)

    # Post-processing (opening + closing)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    fg = cv.morphologyEx(fg, cv.MORPH_OPEN, kernel, iterations=1)
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, kernel, iterations=2)

    return fg


# METRIC (XOR error)
def xor_error(pred_mask, gt_mask):
    """
    XOR error = number of pixels that differ between prediction and GT.
    Lower is better.
    """
    diff = cv.bitwise_xor(pred_mask, gt_mask)
    return int(cv.countNonZero(diff))   


# 5) GRID SEARCH for thresholds
def find_best_thresholds(target_frame, background, gt_mask):
    """
    Search over a reasonable grid of HSV thresholds to find a combination that minimizes the XOR error between the predicted mask and the GT mask.
    Returns: best (th_h, th_s, th_v), best_error
    """

    best = None
    best_err = 10**18

    # Iterate over all the possible values would be too much, so we use a step (5 or 10) in a smaller range to reduce the search space and still cover a wide range of thresholds
    for th_h in range(5, 45, 5):       # 5-45
        for th_s in range(10, 130, 10): # 10-120 
            for th_v in range(10, 130, 10): # 10-120 

                pred = segment_frame_hsv(target_frame, background, th_h, th_s, th_v)
                err = xor_error(pred, gt_mask)

                if err < best_err:
                    best_err = err
                    best = (th_h, th_s, th_v)

    return best, best_err


def main():

    # Build the background model from the background video
    background = get_background_model(FILE_BACKGROUND_PATH)

    frame = cv.imread("assignment_2/data/cam1/img_original.png")
    manual = cv.imread("assignment_2/data/cam1/img_manual.png", cv.IMREAD_GRAYSCALE)

    H, W = background.shape[:2]
    frame = cv.resize(frame, (W, H), interpolation=cv.INTER_LINEAR)
    manual = cv.resize(manual, (W, H), interpolation=cv.INTER_NEAREST)

    # Find the best thresholds using grid search
    best_ths, best_err = find_best_thresholds(frame, background, manual)
    th_h, th_s, th_v = best_ths
    print(f"Best thresholds: H={th_h}, S={th_s}, V={th_v}, with XOR error: {best_err}")

    pred = segment_frame_hsv(frame, background, th_h, th_s, th_v)
    xor = cv.bitwise_xor(pred, manual)

    cv.imshow("Frame", frame)
    cv.imshow("Manual Mask", manual)
    cv.imshow("Predicted Mask", pred)
    cv.imshow("XOR Error Mask", xor)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # Perform background subtraction using the HSV color space
    hsv_background_subtraction(FILE_VIDEO_PATH, background, th_h, th_s, th_v)
    

    
if __name__ == "__main__":
    main()
