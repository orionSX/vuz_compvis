from mymosse import *
import cv2
import os
from time import time
from utils import VideoFileManager

ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0
mouse_pressed = False


def draw_boundingbox(event, x, y, flags, param):
    global ix, iy, cx, cy, w, h, mouse_pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        ix, iy = x, y
        cx, cy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            cx, cy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        w, h = abs(x - ix), abs(y - iy)
        ix, iy = min(ix, x), min(iy, y)


def process_video_with_mosse(video_path, output_dir):
    global ix, iy, cx, cy, w, h, mouse_pressed
    ix, iy, cx, cy = -1, -1, -1, -1
    w, h = 0, 0
    mouse_pressed = False

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("Cannot open video")
        return None

    paused = True
    current_frame = first_frame.copy()

    tracker = None

    cv2.namedWindow("tracking")
    cv2.setMouseCallback("tracking", draw_boundingbox)

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"{video_path.split('mp4')[0].split('\\')[-1]}_mosse_output.mp4"
    )

    out = cv2.VideoWriter(
        output_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h)
    )

    recording = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame = frame.copy()
        else:
            frame = current_frame.copy()

        if paused and tracker is None:
            display = frame.copy()
            if mouse_pressed:
                cv2.rectangle(display, (ix, iy), (cx, cy), (0, 255, 0), 2)
            elif w > 0 and h > 0:
                cv2.rectangle(display, (ix, iy), (ix + w, iy + h), (0, 255, 0), 2)

            cv2.imshow("tracking", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if w > 0 and h > 0:
                    tracker = MOSSETrackerFromOzon()
                    tracker.init([ix, iy, w, h], frame)
                    paused = False
                    recording = True

            elif key == 27:
                return None

            continue

        if tracker is not None:
            bbox = tracker.update(frame)
            x, y, ww, hh = bbox
            cv2.rectangle(frame, (x, y), (x + ww, y + hh), (0, 255, 0), 2)

        if recording:
            out.write(frame)

        cv2.imshow("tracking", frame)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_file


def main():
    input_videos_dir = "videos"
    output_base_dir = "tracking_results"

    video_files = VideoFileManager.get_video_files(input_videos_dir)

    output_dir = VideoFileManager.create_tracker_output_dir(output_base_dir, "my_mosse")

    for i, video_path in enumerate(video_files, 1):
        print(
            f"\nОбработка видео {i}/{len(video_files)}: {os.path.basename(video_path)}"
        )

        video_info = VideoFileManager.get_video_info(video_path)
        if video_info:
            print(f"Файл: {video_info['filename']}, Размер: {video_info['size_mb']} MB")
            process_video_with_mosse(video_path, output_dir)


if __name__ == "__main__":
    main()
