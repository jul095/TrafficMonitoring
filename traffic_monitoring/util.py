#  ****************************************************************************
#  @util.py
#
#  Helper functions
#
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

import cv2
import numpy as np


def calc_center(bbox):
    point_0 = bbox[0:2]
    point_1 = bbox[2:4]

    middlepoint_x = (point_0[0] + point_1[0]) / 2
    middlepoint_y = (point_0[1] + point_1[1]) / 2
    return int(middlepoint_x), int(middlepoint_y)


def convert_polygon_to_mask(polygon):
    width = 1920
    height = 1080
    mask = np.zeros((height, width), np.int8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int)], color=1)
    mask.astype(np.bool)
    return mask


def convert_XYWH_bbox_to_XCYCWH(bbox):
    x, y, w, h = [i for i in bbox]
    return [x + w / 2, y + h / 2, w, h]


def convert_XCYCWH_bbox_to_XYXY(bbox):
    xc, yc, w, h = [i for i in bbox]
    return [xc - (w / 2.), yc - (h / 2), xc + (w / 2.), yc + (h / 2)]


def create_mask_with_points(image_points, image_size=(1080, 1920)):
    image_points = image_points.reshape(-1, 2)
    hull = cv2.convexHull(np.float32(image_points))
    mask = np.zeros(image_size, np.int8)
    cv2.fillConvexPoly(mask, np.array(hull, dtype=np.int), 1, lineType=8, shift=0)
    mask.astype(np.bool)
    return mask
