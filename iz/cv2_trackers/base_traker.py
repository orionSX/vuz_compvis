import cv2
import sys
import os
from time import time


class BaseTracker:
    def __init__(self, video_path=None):
        self.tracker = None
        self.is_initialized = False
        self.bbox = None
        self.video_path = video_path

        self.mouse_pressed = False
        self.ix, self.iy = -1, -1
        self.cx, self.cy = -1, -1
        self.w, self.h = 0, 0

    def init(self, frame, bbox):
        self.tracker = self.create_tracker()
        success = self.tracker.init(frame, bbox)
        self.is_initialized = success
        self.bbox = bbox
        return success

    def update(self, frame):
        if not self.is_initialized or self.tracker is None:
            return False, None

        success, bbox = self.tracker.update(frame)
        self.bbox = bbox if success else None
        return success, bbox

    def get_bbox(self):
        return self.bbox

    def reset(self):
        self.tracker = None
        self.is_initialized = False
        self.bbox = None

    def create_tracker(self):
        raise NotImplementedError("Subclasses must implement create_tracker method")

    def draw_boundingbox(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_pressed = True
            self.ix, self.iy = x, y
            self.cx, self.cy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mouse_pressed:
                self.cx, self.cy = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_pressed = False
            if abs(x - self.ix) > 10 and abs(y - self.iy) > 10:
                self.w, self.h = abs(x - self.ix), abs(y - self.iy)
                self.ix, self.iy = min(x, self.ix), min(y, self.iy)
            else:
                self.w, self.h = 0, 0

    def track_from_video(self, video_path=None, output_path=None):
        if video_path is None:
            video_path = self.video_path

        cap = cv2.VideoCapture(video_path)
        interval = 30

        video_writer = None
        output_video_path = None

        self.mouse_pressed = False
        self.ix, self.iy = -1, -1
        self.cx, self.cy = -1, -1
        self.w, self.h = 0, 0

        tracker = None
        cv2.namedWindow("tracking")
        cv2.setMouseCallback(
            "tracking",
            lambda event, x, y, flags, param: self.draw_boundingbox(
                event, x, y, flags, param
            ),
        )

        ret, first_frame = cap.read()
        if not ret:
            sys.exit("Failed to read video")

        current_frame = first_frame.copy()
        tracking_started = False

        while cap.isOpened():
            if tracking_started:
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                frame = current_frame.copy()

            if not tracking_started:
                display_frame = frame.copy()

                if self.mouse_pressed:
                    cv2.rectangle(
                        display_frame,
                        (self.ix, self.iy),
                        (self.cx, self.cy),
                        (0, 255, 0),
                        2,
                    )

                elif self.w > 0 and self.h > 0:
                    ix_int, iy_int, w_int, h_int = (
                        int(self.ix),
                        int(self.iy),
                        int(self.w),
                        int(self.h),
                    )
                    cv2.rectangle(
                        display_frame,
                        (ix_int, iy_int),
                        (ix_int + w_int, iy_int + h_int),
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("tracking", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    if self.w > 0 and self.h > 0:
                        tracking_started = True
                        ix_int, iy_int, w_int, h_int = (
                            int(self.ix),
                            int(self.iy),
                            int(self.w),
                            int(self.h),
                        )
                        bbox = (ix_int, iy_int, w_int, h_int)
                        self.init(frame, bbox)
                        tracker = self

                        if output_path:
                            output_video_path = output_path
                        else:
                            base_name = os.path.splitext(os.path.basename(video_path))[
                                0
                            ]
                            output_video_path = f"{base_name}_tracked.mp4"

                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(
                            output_video_path, fourcc, fps, (frame_width, frame_height)
                        )

                elif key == 27 or key == ord("q"):
                    break
                continue

            processed_frame = frame.copy()

            if tracker is not None:
                duration = 0.01
                t0 = time()
                success, boundingbox = tracker.update(processed_frame)
                t1 = time()

                if success:
                    boundingbox = list(map(int, boundingbox))
                    x, y, w_track, h_track = boundingbox
                    cv2.rectangle(
                        processed_frame,
                        (x, y),
                        (x + w_track, y + h_track),
                        (0, 255, 0),
                        2,
                    )

                    duration = 0.8 * duration + 0.2 * (t1 - t0)
                    fps = 1 / duration if duration > 0 else 0
                    cv2.putText(
                        processed_frame,
                        f"FPS: {fps:.1f}",
                        (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        processed_frame,
                        "Tracking failure detected",
                        (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

            if video_writer is not None:
                video_writer.write(processed_frame)

            cv2.imshow("tracking", processed_frame)

            c = cv2.waitKey(interval) & 0xFF
            if c == 27 or c == ord("q"):
                break

        cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {output_video_path}")
        cv2.destroyAllWindows()
