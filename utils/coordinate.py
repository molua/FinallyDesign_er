import logging

import carla
import numpy as np


def get_camera_world_view(pixel_cor, K_inverse, c2w, desired_z, sensor_location):
    n = pixel_cor.shape[0]
    v = np.concatenate((pixel_cor, np.array([[1.] * n]).T),axis = 1).T
    F = np.array([[ 0,  1,  0 ], [ 0,  0, -1 ], [ 1,  0,  0 ]], dtype=np.float64)
    F = np.linalg.inv(F)
    to_camera = np.matmul(K_inverse, v)
    to_camera = np.matmul(F, to_camera)
    to_camera = np.concatenate((to_camera, np.array([[1] * n])))
    to_world = np.matmul(c2w, to_camera).T

    vec = to_world[:,:3] - sensor_location.reshape(1,3)
    vec = vec / vec[:,2].reshape(-1,1) * (to_world[:,2].reshape(-1,1) - desired_z)

    return to_world[:,:3] - vec

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    K_inverse = np.linalg.inv(K)
    return K_inverse


def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]

def get_speed(locations, K_inverse, c2w, desired_z, camera_location, fps):
    world_point = get_camera_world_view(locations, K_inverse, c2w, desired_z, camera_location)
    dist = np.linalg.norm(world_point[0] - world_point[1])
    speed = np.around(dist * fps, 2)
    # print(track_id, locations[0], locations[1],  world_point[0], world_point[1], speed)

    return speed

# 计算点到直线的距离
def distance_to_line(x, y, line):
    A, B, C = line
    numerator = np.abs(A*x + B*y + C)
    denominator = np.sqrt(A**2 + B**2)
    distance = numerator / denominator
    return distance

# 最小两个数字的索引
def get_min_two_indices(arr):
    sorted_indices = np.argsort(arr)  # 数值升序排序的索引数组
    min_indices = sorted_indices[:2]  # 前两个索引
    return min_indices

# 判断车所在的车道
def get_lane(x, y, lane_line):
    distance_0 = distance_to_line(x, y, lane_line[0])
    distance_1 = distance_to_line(x, y, lane_line[1])
    distance_2 = distance_to_line(x, y, lane_line[2])
    distance_3 = distance_to_line(x, y, lane_line[3])
    min_indices = get_min_two_indices([distance_0, distance_1, distance_2, distance_3])
    if 0 in min_indices and 1 in min_indices:
        return 0
    elif 1 in min_indices and 2 in min_indices:
        return 1
    elif 2 in min_indices and 3 in min_indices:
        return 2
    else:
        logging.error('get_lane is wrong')
        return -1
