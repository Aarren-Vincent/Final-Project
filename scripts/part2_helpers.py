# part2_helpers.py
import cv2
import os

def read_video_frames(video_path, resize=(320,240), max_frames=None):
    """
    Read frames from video_path, return list of RGB numpy arrays.
    resize: (w,h) target size to keep memory small.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        if resize is not None:
            f = cv2.resize(f, resize)
        # convert BGR->RGB
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames.append(f)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames

def list_videos(folder):
    exts = ('.mp4','.avi','.mov','.mkv')
    videos = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(exts):
                videos.append(os.path.join(root, fn))
    return sorted(videos)
