import logging

import numpy as np
import torch

from utils.coordinate import get_camera_world_view, get_speed, get_lane
from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker


__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=10, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        nn_budget = 100
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    # def update(self, bbox_xywh, confidences, clss, ori_img, c2w, K_inverse, sensor_location, fps, frame, lane_line):
    def update(self, bbox_xywh, confidences, clss, junction):
        ori_img = junction.img_mask
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], clss[i], conf, features[i]) for i, conf in enumerate(
            confidences) if conf > self.min_confidence]
        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []

        # 保存track_id最小的值
        lane_0_track_id_min = 9999999
        lane_0_track_min = None
        lane_0_max_wait = 0

        lane_1_track_id_min = 9999999
        lane_1_track_min = None
        lane_1_max_wait = 0

        lane_2_track_id_min = 9999999
        lane_2_track_min = None
        lane_2_max_wait = 0

        # 记录lane上面的停车数目
        lane_0_stop = 0
        lane_1_stop = 0
        lane_2_stop = 0

        junction.track_num_reset()  # 每次更新的时候都将track_num重置为0
        junction.max_wait_reset()
        junction.all_wait_reset()
        junction.stop_time_reset()

        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            # 计算速度
            x_old = track.last_location[0] + track.last_location[2]/2
            y_old = track.last_location[1] + track.last_location[3]/2
            x_now = track.now_location[0] + track.now_location[2]/2
            y_now = track.now_location[1] + track.now_location[3]/2
            locations = np.array([[x_old, y_old], [x_now, y_now]])

            # print(track.track_id, track.last_location, track.now_location, type(track.last_location), track.last_location.dtype)
            speed = get_speed(locations, junction.K_inverse, junction.c2w, 0, junction.sensor_location, junction.fps)
            lane_id = get_lane(x_now, y_now, junction.lane_line)
            if lane_id == 0:
                junction.lane_0.track_num += 1
                junction.lane_0.all_wait += track.age / junction.fps   # 计算等待时间
                lane_0_max_wait = max(lane_0_max_wait, track.age / junction.fps)
                if speed < 0.99:
                    # lane_0_max_wait = max(lane_0_max_wait, (track.age - track.last_speed_not_zero_time) / junction.fps)
                    # junction.lane_0.all_wait += (track.age - track.last_speed_not_zero_time) / junction.fps   # 计算等待时间
                    junction.lane_0.stop_time += 1
                else:
                    track.last_speed_not_zero_time = track.age

                # if track.track_id < lane_0_track_id_min:
                #     lane_0_track_id_min = track.track_id
                #     lane_0_track_min = track
            elif lane_id == 1:
                junction.lane_1.track_num += 1
                junction.lane_1.all_wait += track.age / junction.fps   # 计算等待时间
                lane_1_max_wait = max(lane_1_max_wait, track.age / junction.fps)
                if speed < 0.99:
                    # lane_1_max_wait = max(lane_1_max_wait, (track.age - track.last_speed_not_zero_time) / junction.fps)
                    # junction.lane_1.all_wait += (track.age - track.last_speed_not_zero_time) / junction.fps   # 计算等待时间
                    junction.lane_1.stop_time += 1
                else:
                    track.last_speed_not_zero_time = track.age

                # if track.track_id < lane_1_track_id_min:
                #     lane_1_track_id_min = track.track_id
                #     lane_1_track_min = track
            elif lane_id == 2:
                junction.lane_2.track_num += 1
                junction.lane_2.all_wait += track.age / junction.fps   # 计算等待时间
                lane_2_max_wait = max(lane_2_max_wait, track.age / junction.fps)
                if speed < 0.99:
                    # lane_2_max_wait = max(lane_2_max_wait, (track.age - track.last_speed_not_zero_time) / junction.fps)
                    # junction.lane_2.all_wait += (track.age - track.last_speed_not_zero_time) / junction.fps   # 计算等待时间
                    junction.lane_2.stop_time += 1
                else:
                    track.last_speed_not_zero_time = track.age

                # if track.track_id < lane_2_track_id_min:
                #     lane_2_track_id_min = track.track_id
                #     lane_2_track_min = track
            outputs.append((x1, y1, x2, y2, track.cls_, track.track_id, speed, lane_id, lane_0_stop, lane_1_stop, lane_2_stop))
        # if lane_0_track_min:
        #     junction.lane_0.max_wait = lane_0_track_min.age / junction.fps
        # if lane_1_track_min:
        #     junction.lane_1.max_wait = lane_1_track_min.age / junction.fps
        # if lane_2_track_min:
        #     junction.lane_2.max_wait = lane_2_track_min.age / junction.fps   # 等待时间需要在每c个时间步进行更新
        junction.lane_0.max_wait = lane_0_max_wait
        junction.lane_1.max_wait = lane_1_max_wait
        junction.lane_2.max_wait = lane_2_max_wait
        return outputs

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        if bbox_tlwh.size(0):
            bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2]/2.
            bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3]/2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x-w/2), 0)
        x2 = min(int(x+w/2), self.width-1)
        y1 = max(int(y-h/2), 0)
        y2 = min(int(y+h/2), self.height-1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width-1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height-1)
        return x1, y1, x2, y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features
