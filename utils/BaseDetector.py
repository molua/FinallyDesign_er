import cv2

from utils.tracker import update_tracker


class baseDet(object):

    def __init__(self):

        self.img_size = 640
        self.threshold = 0.45
        self.stride = 1

    def build_config(self):

        self.faceTracker = {}
        self.faceClasses = {}
        self.faceLocation1 = {}
        self.faceLocation2 = {}
        self.frameCounter = 0
        self.currentCarID = 0
        self.recorded = []

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    # def feedCap(self, im, c2w, K_inverse, sensor_location, fps, lane_line, mask, frame=None,):
    def feedCap(self, junction=None):

        self.frameCounter += 1

        junction.frameCounter += 1

        # im, faces, face_bboxes = update_tracker(self, im, c2w, K_inverse, sensor_location, fps, frame, lane_line, mask)
        im, faces, face_bboxes = update_tracker(self, junction)

        return im

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")
