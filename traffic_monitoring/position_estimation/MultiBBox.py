#  ****************************************************************************
#  @MultiBBox.py
#
#  Representation of a 3D Bounding Box
#
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

import math

import cv2
import numpy as np

from util import create_mask_with_points


class Dim3Bbox:
    """
    This class implements the concept of https://github.com/AubreyC/trajectory-extractor
    and represents the 3D Bounding Box
    """

    def __init__(self, pixel_point, length, width, height, azimuth, camera, detection_id=0):
        self.pixel_point = pixel_point  # center point in Pixel Coordinates
        self.x, self.y, _ = camera.projection_pixel_to_world(self.pixel_point)
        self.length = length
        self.width = width
        self.height = height
        self.azimuth = azimuth
        self.detection_id = detection_id

    def create_mask(self, camera):
        """
        Creates a 2D Segmentation of a 3D Bounding Box. This can be used to compare it with another segmentation
        :param camera:
        :type camera:
        :return: binary mask
        :rtype: 2d numpy array
        """
        self.x, self.y, _ = camera.projection_pixel_to_world(self.pixel_point)
        world_points = self.create_3DBBox()
        image_points = np.array([], np.int)
        for world_point in world_points:
            image_point = camera.projection_world_to_pixel(world_point)
            image_points = np.append(image_points, image_point, axis=0)
        return create_mask_with_points(image_points)

    def project_box_on_image(self, camera):
        """
        calculate a 3d bbox and gives the real 2d pixel coordinates for the corner points of an 3d bbox
        :param camera: Camera object for transformation from 3d in 2d
        :type camera: Camera
        :return: list of corner points
        :rtype: lists
        """
        world_points = self.create_3DBBox()

        image_points = np.array([], np.int)
        for world_point in world_points:
            image_point = camera.projection_world_to_pixel(world_point)
            image_points = np.append(image_points, image_point, axis=0)

        return image_points

    def create_3DBBox(self):
        """
        Creates the corner points of a 3d bounding box in world coordinates
        :return: list of corner points
        :rtype: list
        """
        center = np.array([self.x, self.y, 0])
        center.shape = (3, 1)

        R_z = np.array([[math.cos(self.azimuth), math.sin(self.azimuth), 0],
                        [-math.sin(self.azimuth), math.cos(self.azimuth), 0],
                        [0, 0, 1]
                        ]).transpose()

        #  bottom up left
        tr = np.array([self.length / 2, -self.width / 2, 0])
        tr.shape = (3, 1)
        pt1 = center + R_z.dot(tr)

        # bottom up right
        tr = np.array([self.length / 2, self.width / 2, 0])
        tr.shape = (3, 1)
        pt2 = center + R_z.dot(tr)

        #  bottom down left
        tr = np.array([-self.length / 2, -self.width / 2, 0])
        tr.shape = (3, 1)
        pt3 = center + R_z.dot(tr)

        #  bottom down right
        tr = np.array([-self.length / 2, self.width / 2, 0])
        tr.shape = (3, 1)
        pt4 = center + R_z.dot(tr)

        #  top up left
        tr = np.array([self.length / 2, -self.width / 2, self.height])
        tr.shape = (3, 1)
        pt5 = center + R_z.dot(tr)

        # top up right
        tr = np.array([self.length / 2, self.width / 2, self.height])
        tr.shape = (3, 1)
        pt6 = center + R_z.dot(tr)

        #  top down left
        tr = np.array([-self.length / 2, -self.width / 2, self.height])
        tr.shape = (3, 1)
        pt7 = center + R_z.dot(tr)

        #  top down right
        tr = np.array([-self.length / 2, self.width / 2, self.height])
        tr.shape = (3, 1)
        pt8 = center + R_z.dot(tr)

        list_corner_points = [pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8]
        return list_corner_points

    def draw_on_frame(self, frame, camera, color=(204, 102, 255), thickness=2):
        """
        draw the 3d bounding box on a frame
        :param frame: image frame
        :type frame: frame
        :param camera: camera object for transformation in world space and back
        :type camera: Camera
        :param color: Color of the lines and points
        :type color: rgb tuple
        :param thickness: Thickness of the line
        :type thickness: int
        :return: image frame
        :rtype: frame
        """

        #  Project point on image plane
        image_points = self.project_box_on_image(camera)

        point_image_tuple = []

        for point in image_points.reshape(-1, 2):
            tuple_point = (int(point[0]), int(point[1]))
            point_image_tuple.append(tuple_point)
            cv2.circle(frame, tuple_point, 2, color, -1)

        #  draw lines with the corner points
        cv2.line(frame, point_image_tuple[0], point_image_tuple[1], color, thickness)
        cv2.line(frame, point_image_tuple[0], point_image_tuple[2], color, thickness)
        cv2.line(frame, point_image_tuple[1], point_image_tuple[3], color, thickness)
        cv2.line(frame, point_image_tuple[2], point_image_tuple[3], color, thickness)

        cv2.line(frame, point_image_tuple[4], point_image_tuple[5], color, thickness)
        cv2.line(frame, point_image_tuple[4], point_image_tuple[6], color, thickness)
        cv2.line(frame, point_image_tuple[5], point_image_tuple[7], color, thickness)
        cv2.line(frame, point_image_tuple[6], point_image_tuple[7], color, thickness)

        cv2.line(frame, point_image_tuple[0], point_image_tuple[4], color, thickness)
        cv2.line(frame, point_image_tuple[1], point_image_tuple[5], color, thickness)
        cv2.line(frame, point_image_tuple[2], point_image_tuple[6], color, thickness)
        cv2.line(frame, point_image_tuple[3], point_image_tuple[7], color, thickness)

        return frame
