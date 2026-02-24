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

    Returns: a binary mask of the foreground objects.
    """

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Can't open the video: {video_path}")

    # Convert background model to HSV
    hsv_background = cv.cvtColor(background_model, cv.COLOR_BGR2HSV)
    # Split the background model into H, S, and V channels
    hb, sb, vb = cv.split(hsv_background) 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv_frame)
        
        # hue difference should consider the circular nature of hue values (0-179 in OpenCV) ??
        diff_h = cv.absdiff(h, hb)
        diff_h = np.minimum(diff_h, 180 - diff_h) # circular hue difference
        diff_s = cv.absdiff(s, sb)
        diff_v = cv.absdiff(v, vb)

        # Convert differences to binary masks based on thresholds
        mask_h = (diff_h > th_h).astype(np.uint8) * 255 
        mask_s = (diff_s > th_s).astype(np.uint8) * 255     
        mask_v = (diff_v > th_v).astype(np.uint8) * 255

        # We combine the masks using a logical OR op to get the final binary foreground mask
        foreground_mask = cv.bitwise_or(mask_h, cv.bitwise_or(mask_s, mask_v))


        # Display results
        cv.imshow('Frame', frame)
        cv.imshow('Foreground Mask', foreground_mask)

        if cv.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

    
def main():

    # Build the background model from the background video
    background = get_background_model(FILE_BACKGROUND_PATH)

    # Perform background subtraction using the HSV color space
    hsv_background_subtraction(FILE_VIDEO_PATH, background)

    
if __name__ == "__main__":
    main()
