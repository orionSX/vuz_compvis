import cv2
from . import BaseTracker

class KCFTracker(BaseTracker):
    def create_tracker(self):
        return cv2.legacy.TrackerKCF_create()
