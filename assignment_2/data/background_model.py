import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import argparse


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
FILE_BACKGROUND_PATH = "assignment_2/data/cam1/background.avi"
FILE_VIDEO_PATH = "assignment_2/data/cam1/video.avi"

def get_background_model(video_path: str, num_frames: int = 30):
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
        raise ValueError("No frames were read from the video.")

    background_model = np.mean(frames, axis=0).astype(np.uint8)
    return background_model

def start_substracting(video_path, background_model):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Can't open the video: {video_path}")

    ret, background_model = cap.read()
    if not ret:
        raise ValueError("Can't read the first frame of the video.")
    background_model = background_model.astype(np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_float = frame.astype(np.float32)
        diff = cv.absdiff(frame_float, background_model)
        gray_diff = cv.cvtColor(diff.astype(np.uint8), cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(gray_diff, 30, 255, cv.THRESH_BINARY)

        # Update the background model
        alpha = 0.05
        background_model = (1 - alpha) * background_model + alpha * frame_float

        # Display results
        cv.imshow('Frame', frame)
        cv.imshow('Mask', mask)

        if cv.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def main():
    cap = cv.VideoCapture(FILE_VIDEO_PATH)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        fgMask = cv.createBackgroundSubtractorMOG2().apply(frame)
         
        cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        
        cv.imshow('Frame', frame)
        cv.imshow('FG Mask', fgMask)
        if cv.waitKey(30) & 0xFF == ord('q'):
            break

    
if __name__ == "__main__":
    main()
