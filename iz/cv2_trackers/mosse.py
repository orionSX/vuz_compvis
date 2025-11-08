import cv2
from . import BaseTracker

class MOSSETracker(BaseTracker):
    def create_tracker(self):
        return cv2.legacy.TrackerMOSSE_create()
