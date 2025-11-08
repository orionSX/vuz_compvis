import cv2
from . import BaseTracker

class CSRTTracker(BaseTracker):
    def create_tracker(self):
        return cv2.legacy.TrackerCSRT_create()
