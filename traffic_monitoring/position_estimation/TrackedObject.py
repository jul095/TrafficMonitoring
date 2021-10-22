#  ****************************************************************************
#  @TrackedObject.py
#
#  Estimation of the movement direction to get front and rear information of each vehicle
#
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

import math

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class TrackedObject:
    """

    """
    eval_interval = 10

    def __init__(self, track_id, class_id):
        self.track_id = track_id
        self.class_id = class_id
        self.track_count = 0
        self.bottom_plates = []
        self.center_points = []
        self.velocity_reference_points = []
        self.movement_vector = None
        self.old_movement_vector = None
        self.movement_threshold = 0.15

        self.velocity_frame_window = 5
        self.FPS_OF_RECORDING = 30.

    def get_all_trajectory_points(self):
        return self.center_points

    def add_bbox_and_center(self, bbox, center_point):
        self.bottom_plates.append(bbox)
        self.center_points.append(center_point)
        self.velocity_reference_points.append(center_point)
        self.track_count += 1
        return self.calculate_velocities()

    def add_bottom_plate_and_center(self, bottom_plate, center_point, velocity_reference_point):
        self.bottom_plates.append(bottom_plate)
        self.center_points.append(center_point)
        self.velocity_reference_points.append(velocity_reference_point)
        self.track_count += 1
        self.calculate_movement_vector()
        return self.get_position_of_vehicle(), self.calculate_velocities()

    def calculate_movement_vector(self):
        if (self.track_count % self.eval_interval) == 0 or self.track_count == 5:
            if self.track_count > 5:
                start = self.center_points[self.track_count - 4]
            else:
                start = self.center_points[0]
            end = self.center_points[self.track_count - 1]
            movement_vector = end - start
            if np.linalg.norm(movement_vector[0]) > self.movement_threshold or np.linalg.norm(
                    movement_vector[1]) > self.movement_threshold:
                self.movement_vector = movement_vector

    def get_position_of_vehicle(self):
        #  idee: nur den vektor puffern und die kanten jedes mal mit cosine prüfen

        if self.movement_vector is not None:
            end_bottom_plate = self.bottom_plates[-1]
            #  vector0_length = np.linalg.norm(end_bottom_plate[3] - end_bottom_plate[2])
            #  vector1_length = np.linalg.norm(end_bottom_plate[2] - end_bottom_plate[1])

            #  if vector0_length <= vector1_length:
            #    direction_vector = end_bottom_plate[2] - end_bottom_plate[1]
            #  else:
            #    direction_vector = end_bottom_plate[3] - end_bottom_plate[2]
            direction_vector = end_bottom_plate[1] - end_bottom_plate[0]

            direction_vector2 = end_bottom_plate[2] - end_bottom_plate[0]

            if np.linalg.norm(self.movement_vector[0]) > self.movement_threshold or np.linalg.norm(
                    self.movement_vector[1]) > self.movement_threshold:
                # print('direction_vector', direction_vector)
                similarity_vector = cosine_similarity(direction_vector.reshape(1, -1),
                                                      self.movement_vector.reshape(1, -1))
                similarity_vector2 = cosine_similarity(direction_vector2.reshape(1, -1),
                                                       self.movement_vector.reshape(1, -1))

                if similarity_vector > 0.7:
                    #  end_bottom_plate[1] is front of the car
                    self.front = end_bottom_plate[1]
                    self.back = end_bottom_plate[0]
                    small_edge = end_bottom_plate[3] - end_bottom_plate[1]
                    return self.estimate_gps_sensor_position(end_bottom_plate, small_edge)
                elif similarity_vector < - 0.7:
                    #  end_bottom_plate[0] is front of the car
                    self.front = end_bottom_plate[0]
                    self.back = end_bottom_plate[1]
                    small_edge = end_bottom_plate[3] - end_bottom_plate[1]
                    return self.estimate_gps_sensor_position(end_bottom_plate, small_edge)

                if similarity_vector2 > 0.8:
                    #  end_bottom_plate[1] is front of the car
                    self.front = end_bottom_plate[2]
                    self.back = end_bottom_plate[0]
                    small_edge = end_bottom_plate[1] - end_bottom_plate[0]
                    return self.estimate_gps_sensor_position(end_bottom_plate, small_edge)
                elif similarity_vector2 < - 0.8:
                    #  end_bottom_plate[0] is front of the car
                    self.front = end_bottom_plate[0]
                    self.back = end_bottom_plate[2]
                    small_edge = end_bottom_plate[1] - end_bottom_plate[0]
                    return self.estimate_gps_sensor_position(end_bottom_plate, small_edge)

        # if self.current_point is not None:
        #    return self.current_point
        return self.center_points[-1]

    def estimate_gps_sensor_position(self, end_bottom_plate, small_edge):
        self.current_point = ((2 / 3.) * (self.back - self.front)) + self.front + (1 / 2.) * small_edge
        return self.current_point

    def calculate_velocities(self):
        if self.velocity_frame_window < self.track_count:
            prev_point = self.velocity_reference_points[self.track_count - 1 - self.velocity_frame_window]
            # prev_point = self.velocity_reference_points[self.track_count - 2]
            curr_point = self.velocity_reference_points[self.track_count - 1]
            velo_in_km_h = math.sqrt(
                math.pow(prev_point[0] - curr_point[0], 2) + math.pow(prev_point[1] - curr_point[1], 2)) * 3.6 / (
                                   self.velocity_frame_window / self.FPS_OF_RECORDING)
            return velo_in_km_h
        return 0.0
