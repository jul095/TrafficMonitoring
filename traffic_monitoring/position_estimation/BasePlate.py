#  ****************************************************************************
#  @BasePlate.py
#
#  Representation of the vehicle base plate
#  Drawing Options and height calculation of the vehicles
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

import math

import cv2
import numpy as np


class BasePlate:
    """
    Represents, create and draw a bottom plate of an vehicle
    """

    def __init__(self, bottom_edge, top_edge, camera, box, is_square, vehicle_height=1.5, vehicle_width=2.0):
        self.bottom_edge = bottom_edge
        self.top_edge = top_edge
        self.vehicle_height = vehicle_height
        self.vehicle_width = vehicle_width
        self.camera = camera
        self.box = box
        self.is_square = is_square

    def project_box_on_image(self):
        """
        calculate a 3d bbox and gives the real 2d pixel coordinates for the corner points of an 3d bbox
        :return: list of corner points
        :rtype: lists
        """
        world_points = self.create_3DBBox()

        image_points = np.array([], np.int)
        for world_point in world_points:
            image_point = self.camera.projection_world_to_pixel(world_point)
            image_points = np.append(image_points, image_point, axis=0)
        return image_points

    def calculate_vehicle_height(self, bottom_point, top_point):
        """
        calculate vehicle height based on vehicle width
        """
        length_vector = np.linalg.norm(top_point - bottom_point)
        if length_vector < 0.5:
            return self.vehicle_height
        return np.sqrt(np.power(length_vector, 2) - np.power(self.vehicle_width, 2))

    def create_3DBBox(self):
        """
        create a 3D Bbox with Rotation and Translation
        based on a predefined vehicle width and the top and bottom edge

        Returns a point list which represents a cuboid in following order

                   7--------5
                  /|       /|
                 / |      / |
                6  |     4  |
                |  3-----|  1
                | /      | /
                |/       |/
                2--------0

        """
        angle = np.deg2rad(90)
        r_z = np.array([[math.cos(angle), math.sin(angle), 0],
                        [-math.sin(angle), math.cos(angle), 0],
                        [0, 0, 1]
                        ]).transpose()

        if not self.is_square:
            # create 3d bbox with bottom and top edge
            vertice0_bottom = self.camera.projection_pixel_to_world(
                np.array([self.bottom_edge[0][0], self.bottom_edge[0][1], 1]).reshape((3, 1)))
            vertice1_bottom = self.camera.projection_pixel_to_world(
                np.array([self.bottom_edge[1][0], self.bottom_edge[1][1], 1]).reshape((3, 1)))

            vertice0_top = self.camera.projection_pixel_to_world(
                np.array([self.top_edge[0][0], self.top_edge[0][1], 1]).reshape((3, 1)))
            vertice1_top = self.camera.projection_pixel_to_world(
                np.array([self.top_edge[1][0], self.top_edge[1][1], 1]).reshape((3, 1)))

            # vehicle_height = self.calculate_vehicle_height(vertice0_bottom, vertice1_top)

            z = np.float(-self.vehicle_height)  # fixed vehicle size

            pt0 = np.array([vertice0_bottom[0], vertice0_bottom[1], 0])
            pt1 = np.array([vertice1_bottom[0], vertice1_bottom[1], 0])

            vector_bottom_line = np.array(
                [vertice1_bottom[0] - vertice0_bottom[0], vertice1_bottom[1] - vertice0_bottom[1]])
            vector_bottom_line_distance = np.linalg.norm(vector_bottom_line)
            vector_bottom_line = vector_bottom_line / vector_bottom_line_distance
            vector_bottom_line = np.array([vector_bottom_line[0], vector_bottom_line[1], 1])
            vector_bottom_line.shape = (3, 1)

            vector_bottom_line = r_z.dot(vector_bottom_line).reshape((1, 3))[0]

            own_left_bottom_point_1 = vertice0_bottom + self.vehicle_width * vector_bottom_line
            own_left_bottom_point_2 = vertice1_bottom + self.vehicle_width * vector_bottom_line

            pt2 = np.array([own_left_bottom_point_1[0], own_left_bottom_point_1[1], 0])
            pt3 = np.array([own_left_bottom_point_2[0], own_left_bottom_point_2[1], 0])

            pt4 = np.array([vertice0_bottom[0], vertice0_bottom[1], (-1.) * z])
            pt5 = np.array([vertice1_bottom[0], vertice1_bottom[1], (-1.) * z])
            pt6 = np.array([vertice0_top[0], vertice0_top[1], 0])
            pt7 = np.array([vertice1_top[0], vertice1_top[1], 0])
        else:
            # In case of a squared bottom plate there is no difference
            # between bottom and top edge so the rotated rectangle is the base plate

            vertice0 = self.camera.projection_pixel_to_world(
                np.array([self.box[0][0], self.box[0][1], 1]).reshape((3, 1)))
            vertice1 = self.camera.projection_pixel_to_world(
                np.array([self.box[1][0], self.box[1][1], 1]).reshape((3, 1)))
            vertice2 = self.camera.projection_pixel_to_world(
                np.array([self.box[2][0], self.box[2][1], 1]).reshape((3, 1)))
            vertice3 = self.camera.projection_pixel_to_world(
                np.array([self.box[3][0], self.box[3][1], 1]).reshape((3, 1)))

            z = self.vehicle_height
            pt0 = np.array([vertice0[0], vertice0[1], 0])
            pt1 = np.array([vertice1[0], vertice1[1], 0])
            pt2 = np.array([vertice3[0], vertice3[1], 0])
            pt3 = np.array([vertice2[0], vertice2[1], 0])

            pt4 = np.array([vertice0[0], vertice0[1], (-1.) * z])
            pt5 = np.array([vertice1[0], vertice1[1], (-1.) * z])
            pt6 = np.array([vertice3[0], vertice3[1], (-1.) * z])
            pt7 = np.array([vertice2[0], vertice2[1], (-1.) * z])

        list_corner_points = [pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7]

        return list_corner_points

    def get_center_base_point_world(self):
        """
        returns the center of the base plate of the vehicle in world coordinates
        """
        list_corner_points = self.create_3DBBox()
        corner_points_bottom_plate = list_corner_points[0:4]
        vector_rectangle = np.array(corner_points_bottom_plate[3] - corner_points_bottom_plate[0])
        middle_point = vector_rectangle * 0.5 + corner_points_bottom_plate[0]
        return middle_point, corner_points_bottom_plate

    def get_center_base_point_pixel(self):
        """
        return the center of the base plate of the vehicle in pixel coordinates
        """
        middle_point = self.get_center_base_point_world()
        image_point = self.camera.projection_world_to_pixel(middle_point)
        return image_point

    def draw_on_frame(self, frame, color=(255, 255, 0), thickness=2):
        image_points = self.project_box_on_image()

        point_image_tuple = []
        for point in image_points.reshape(-1, 2):
            tuple_point = (int(point[0]), int(point[1]))
            point_image_tuple.append(tuple_point)

        #  draw lines with the corner points
        #
        #      3---------1
        #     /         /
        #    /         /
        #   /         /
        #  2---------0
        #

        cv2.line(frame, point_image_tuple[0], point_image_tuple[1], color, thickness)
        cv2.line(frame, point_image_tuple[0], point_image_tuple[2], color, thickness)
        cv2.line(frame, point_image_tuple[1], point_image_tuple[3], color, thickness)
        cv2.line(frame, point_image_tuple[2], point_image_tuple[3], color, thickness)

        # world_middle_point = self.get_center_bottom_point_world()
        # image_point = self.camera.projection_world_to_pixel(world_middle_point)
        # cv2.circle(frame, (image_point[0], image_point[1]), 2, (0, 255, 0), -1)
        return frame
