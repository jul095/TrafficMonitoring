#  ****************************************************************************
#  @EstimateVehicleBasePlate.py
#
#  Estimation of the Base Plate for each vehicle
#  based on rotated Bounding Box
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

import os

import cv2
import numpy as np
from scipy.spatial import ConvexHull

from geomapping import CameraCalibration
from position_estimation.BasePlate import BasePlate
from read_dataset.ReadCOCODataset import COCOFile
from vis.Visualizer import Visualizer


class EstimateVehicleBasePlate:
    """
    Algorithms to find the Base Plate of an vehicle
    for finding the optimal reference point of an vehicle
    """
    def __init__(self, passpoints_file_path):
        self.visualizer = Visualizer()
        self.camera = CameraCalibration(passpoints_file_path)

    @DeprecationWarning
    def minimum_bounding_rectangle(self, points):
        """
        Find the smallest bounding rectangle for a set of points.
        Returns a set of points representing the corners of the bounding box.

        :param points: an nx2 matrix of coordinates
        :rval: an nx2 matrix of coordinates
        """

        pi2 = np.pi / 2.

        # get the convex hull for the points
        hull_points = points[ConvexHull(points).vertices]

        # calculate edge angles
        edges = np.zeros((len(hull_points) - 1, 2))

        # print("hull_points_1", hull_points[1:], "hull_points_-1", hull_points[:-1])
        edges = hull_points[1:] - hull_points[:-1]

        angles = np.zeros((len(edges)))
        angles = np.arctan2(edges[:, 1], edges[:, 0])

        angles = np.abs(np.mod(angles, pi2))
        angles = np.unique(angles)

        # find rotation matrices
        # XXX both work
        rotations = np.vstack([
            np.cos(angles),
            np.cos(angles - pi2),
            np.cos(angles + pi2),
            np.cos(angles)]).T
        # rotations = np.vstack([
        #   np.cos(angles),
        #   -np.sin(angles),
        #   np.sin(angles),
        #   np.cos(angles)]).T
        rotations = rotations.reshape((-1, 2, 2))

        # apply rotations to the hull
        rot_points = np.dot(rotations, hull_points.T)
        # TODO maybe here instead of max area max width
        # find the bounding points

        min_x = np.nanmin(rot_points[:, 0], axis=1)
        max_x = np.nanmax(rot_points[:, 0], axis=1)
        min_y = np.nanmin(rot_points[:, 1], axis=1)
        max_y = np.nanmax(rot_points[:, 1], axis=1)

        # find the box with the best area

        distance_scores = np.zeros((len(min_y), 1))
        area_scores = np.zeros((len(min_y), 1))
        for idx, (x0, y0, x1, y1) in enumerate(zip(min_x, min_y, max_x, max_y)):
            # print(x0, y0, x1, y1)
            point0 = np.array([x0, y0])
            point1 = np.array([x0, y1])
            distance = np.linalg.norm(point1 - point0)
            area = (x1 - x0) * (y1 - y0)

            distance_scores[idx] = distance
            area_scores[idx] = area

            # print("distance: ", distance)

        distance_scores_ids = np.argsort(distance_scores)
        area_scores_ids = np.argsort(area_scores[:, 0])

        max_distance = 0
        max_area = np.infty
        max_of_both = 0
        current_index = 4

        current_index = area_scores_ids[0]

        # distance = np.linalg.norm((min_x, min_y) - (max_x, max_y))

        # areas = (max_x - min_x) * (max_y - min_y)
        # best_idx = np.argmin(areas)

        # return the best box
        x1 = max_x[current_index]
        x2 = min_x[current_index]
        y1 = max_y[current_index]
        y2 = min_y[current_index]
        r = rotations[current_index]

        rval = np.zeros((4, 2))
        rval[0] = np.dot([x1, y2], r)
        rval[1] = np.dot([x2, y2], r)
        rval[2] = np.dot([x2, y1], r)
        rval[3] = np.dot([x1, y1], r)

        return rval

    def find_bottom_top_edge(self, box):
        """
        Use the rotated bounding box to select two edges, the top and bottom edge of the vehicle
        In case of an squared bbox, this will be used as base plate
        """
        bottom_edge = np.zeros((2, 2))
        top_edge = np.zeros((2, 2))
        unique_x, _ = np.unique(box[:, 0], return_counts=True)
        unique_y, _ = np.unique(box[:, 1], return_counts=True)

        is_square = False

        if len(box) == 4 and len(unique_x) > 2 and len(unique_y) > 2:
            # Sort points by y coordinate
            sort_order = box[:, 1].argsort()[::-1]
            # Select highest y coordinate
            index_highest_y_coordinate = sort_order[0]
            bottom_vertice_candidate_0 = box[(index_highest_y_coordinate - 1) % 4]
            bottom_vertice_candidate_1 = box[(index_highest_y_coordinate + 1) % 4]
            top_edge[0] = box[(index_highest_y_coordinate + 2) % 4]

            sort_by_y_desc = box[sort_order]
            #  This point is guranted part of the bottom edge of a object
            bottom_vertice_0 = sort_by_y_desc[0]
            bottom_edge[0] = bottom_vertice_0
            candidate_length_0 = np.linalg.norm(bottom_vertice_0 - bottom_vertice_candidate_0)
            candidate_length_1 = np.linalg.norm(bottom_vertice_0 - bottom_vertice_candidate_1)
            if candidate_length_0 > candidate_length_1:
                bottom_edge[1] = bottom_vertice_candidate_0
                top_edge[1] = bottom_vertice_candidate_1
            else:
                bottom_edge[1] = bottom_vertice_candidate_1
                top_edge[1] = bottom_vertice_candidate_0
        else:
            # This case will be called if the rotated bbox is a square
            is_square = True

            sorted_box = box[np.lexsort((box[:, 0], box[:, 1]))][::-1]

            candidate_length_0 = np.linalg.norm(sorted_box[0] - sorted_box[1])
            candidate_length_1 = np.linalg.norm(sorted_box[1] - sorted_box[3])

            if candidate_length_0 < candidate_length_1:
                top_edge[0] = sorted_box[1]
                top_edge[1] = sorted_box[3]
                bottom_edge[0] = sorted_box[2]
                bottom_edge[1] = sorted_box[0]
            else:
                top_edge[0] = sorted_box[0]
                top_edge[1] = sorted_box[1]
                bottom_edge[0] = sorted_box[3]
                bottom_edge[1] = sorted_box[2]

            if len(unique_y) == 2:
                # Sort by y-coordinates
                sort_order = box[:, 1].argsort()[::-1]
                # select first
                index_highest_y_coordinate = sort_order[0]
                sort_by_y_desc = box[sort_order]
                #  This point is guaranteed part of the bottom edge of a object
                bottom_vertice_0 = sort_by_y_desc[0]
                vertice1 = sort_by_y_desc[1]
                vertice2 = sort_by_y_desc[2]
                vertice3 = sort_by_y_desc[3]
                if bottom_vertice_0[0] == vertice2[0]:
                    length_vector_1 = np.linalg.norm(vertice2 - bottom_vertice_0)
                else:
                    length_vector_1 = np.linalg.norm(vertice3 - bottom_vertice_0)

                length_y_vector = np.linalg.norm(vertice1 - bottom_vertice_0)

                if length_y_vector > length_vector_1:
                    if bottom_vertice_0[0] < vertice1[0]:
                        bottom_edge[0] = vertice1
                        bottom_edge[1] = bottom_vertice_0
                    else:
                        bottom_edge[0] = bottom_vertice_0
                        bottom_edge[1] = vertice1
                    top_edge[0] = vertice2
                    top_edge[1] = vertice3
                    is_square = False

        if bottom_edge[0][0] >= bottom_edge[1][0] or bottom_edge[0][1] <= bottom_edge[1][1]:
            bottom_edge = bottom_edge[::-1]

        return np.asarray(bottom_edge, dtype=np.float32), np.asarray(top_edge, dtype=np.float32), is_square

    def find_rotated_bbox_and_base_plate(self, frame, mask_polygon, category_name):
        """
        At first find the rotated bounding box based on the segmentation of a vehicle
        returns a base plate object based on the category (car, truck, transporter)

        """
        convex_hull = cv2.convexHull(np.float32(mask_polygon))
        rect = cv2.minAreaRect(np.float32(convex_hull))
        box = cv2.boxPoints(rect)
        box = np.asarray(box)
        bottom_edge, top_edge, is_square = self.find_bottom_top_edge(box)

        # Try to calculate the rotated BBox based on a ellipse which is covering the segmentation of the object

        # minEllipse = cv2.fitEllipse(np.float32(convex_hull))
        # print(minEllipse)
        # white_img = np.zeros((1080,1920,1), np.uint8)
        # cv2.ellipse(white_img, minEllipse, (255,0,0))
        # white_img.astype(np.bool)
        # polygons_ellipse = np.array(Mask(white_img).polygons()[0])
        # polygon_ellipse = polygons_ellipse.reshape(-1,2)
        # rect = cv2.minAreaRect(polygon_ellipse)
        # print(rect)

        # box = self.minimum_bounding_rectangle(mask_polygon)
        # box = np.int0(box)
        # print("track_id and rotated bbox object", box)

        vehicle_height = 0
        vehicle_width = 0
        if category_name == 'car':
            vehicle_height = 1.50
            vehicle_width = 1.80
        elif category_name == 'truck':
            vehicle_height = 4.00
            vehicle_width = 2.55
        elif category_name == 'transporter':
            vehicle_height = 2.45
            vehicle_width = 2.0
        else:
            vehicle_width = 0.5
            vehicle_height = 1.5

        base_plate = BasePlate(bottom_edge, top_edge, self.camera, box, is_square, vehicle_height, vehicle_width)
        base_plate.draw_on_frame(frame)

        # visual statements for debugging purpose

        # self.visualizer.draw_rotated_bbox(frame, box)
        # cv2.line(frame, (bottom_edge[0][0], bottom_edge[0][1]), (bottom_edge[1][0], bottom_edge[1][1]), (255, 255, 0),
        #         4)
        # cv2.line(frame, (top_edge[0][0], top_edge[0][1]), (top_edge[1][0], top_edge[1][1]), (0, 255, 0),
        #         4)

        return base_plate

    def find_base_plate_and_center_world(self, frame, mask_polygon, category_name):
        bbox3d = self.find_rotated_bbox_and_base_plate(frame, mask_polygon, category_name)
        center, bottom_plate = bbox3d.get_center_base_point_world()

        return center, bottom_plate

    def estimate_3d_bbox_on_all_annotations(self, dataset, coco):
        """
        Method for testing the hole Base Plate Algorithm standalone without segmentation
        instead a Ground Truth Dataset will be used
        """
        for frame_id, data in enumerate(dataset):
            frame = cv2.imread(data["file_name"])
            for elem in data.get("annotations"):
                gt_class_id = elem.get('category_id')
                gt_track_id = elem.get('track_id')
                bbox = elem.get('bbox')
                polygon_in_one_row = np.asarray(elem.get('segmentation'))[0]
                polygon = polygon_in_one_row.reshape(-1, 2)
                label_text = "Class ID: %i Track ID: %i" % (gt_class_id, gt_track_id)
                frame = self.visualizer.draw_mask_with_polygon(frame, polygon_in_one_row, bbox, gt_class_id, label_text)
                category_name = coco.get_category_name_by_new_id(gt_class_id)
                self.find_base_plate_and_center_world(frame, polygon, category_name)

            cv2.imshow("Show test image", frame)
            cv2.waitKey(0)


if __name__ == '__main__':
    dataset_path = os.path.join(os.path.dirname(__file__), '../..', 'data/dataset',
                                'training_local')
    folder_of_datasets = os.listdir(dataset_path)
    coco_file = COCOFile(os.path.join(dataset_path, folder_of_datasets[0]))
    dataset = coco_file.get_detectron2_dataset(True)
    estimate3DBBox = EstimateVehicleBasePlate()
    estimate3DBBox.estimate_3d_bbox_on_all_annotations(dataset, coco_file)
