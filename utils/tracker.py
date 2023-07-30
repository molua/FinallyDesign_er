from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2

from utils.coordinate import get_lane

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)


def plot_bboxes(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    image_0 = image.copy()
    # image_0 = image.deepcopy()
    image_1 = None

    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    for i, (x1, y1, x2, y2, cls_id, pos_id, speed, lane_id, lane_0_stop, lane_1_stop, lane_2_stop) in enumerate(bboxes):
        if cls_id in ['person']:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image_0, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 7, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        image_0 = image_0.copy()
        # image_0 = image_0.deepcopy()
        cv2.rectangle(image_0, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image_0, '{}-{}'.format(pos_id, round(speed, 1)), (c1[0], c1[1] - 2), 0, tl / 9,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        image_1 = image_0.copy()
        # if i == len(bboxes):
        cv2.putText(image_1, 'left:{}  middle:{}  right:{}'.format(lane_0_stop, lane_1_stop, lane_2_stop), (10, t_size[1] + 10), 0, tl / 5, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    # return image_0
    return image_1


# def update_tracker(target_detector, image, c2w, K_inverse, sensor_location, fps, frame, lane_line, mask):
def update_tracker(target_detector, junction):

    new_faces = []
    _, bboxes = target_detector.detect(junction.img_mask)

    bbox_xywh = []
    confs = []
    clss = []

    junction.vehicle_num_reset()

    outputs = []

    for x1, y1, x2, y2, cls_id, conf in bboxes:
        if ((x2 - x1) < 200) and ((y2 - y1) < 200):
            obj = [
                int((x1+x2)/2), int((y1+y2)/2),
                x2-x1, y2-y1
            ]
            bbox_xywh.append(obj)
            confs.append(conf)
            clss.append(cls_id)

            lane_id = get_lane(obj[0], obj[1], junction.lane_line)
            if lane_id == 0:
                junction.lane_0.vehicle_num += 1
            elif lane_id == 1:
                junction.lane_1.vehicle_num += 1
            elif lane_id == 2:
                junction.lane_2.vehicle_num += 1

    xywhs = torch.Tensor(bbox_xywh)
    confss = torch.Tensor(confs)

    outputs = junction.deepsort.update(xywhs, confss, clss, junction)
    # outputs = deepsort.update(xywhs, confss, clss, image, c2w, K_inverse, sensor_location, fps, frame, lane_line)

    bboxes2draw = []
    face_bboxes = []
    current_ids = []
    for value in list(outputs):
        x1, y1, x2, y2, cls_, track_id, speed, lane_id, lane_0_stop, lane_1_stop, lane_2_stop = value
        bboxes2draw.append(
            (x1, y1, x2, y2, cls_, track_id, speed, lane_id, lane_0_stop, lane_1_stop, lane_2_stop)
        )
        current_ids.append(track_id)

    image = plot_bboxes(junction.img, bboxes2draw)

    return image, new_faces, face_bboxes
    # return junction.img, new_faces, face_bboxes
