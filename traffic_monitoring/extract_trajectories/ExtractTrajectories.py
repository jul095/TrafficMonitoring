#  ****************************************************************************
#  @ExtractTrajectories.py
#
#  Write the Trajectories to an CSV file
#
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

import csv

from numpy import NaN, array


class ExtractTrajectories:
    """
    Store the Trajectories in a csv file
    """
    def __init__(self, file_name):
        self.file_name = file_name

        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["frame_id", "class_id", "track_id", "velocity", 
            "coordinate_world_x_opt", "coordinate_world_y_opt", 
            "coordinate_world_x", "coordinate_world_y", 
            "bottom_plate_1_x", "bottom_plate_1_y", "bottom_plate_1_z",
            "bottom_plate_2_x", "bottom_plate_2_y", "bottom_plate_2_z",
            "bottom_plate_3_x", "bottom_plate_3_y", "bottom_plate_3_z", 
            "bottom_plate_4_x", "bottom_plate_4_y", "bottom_plate_4_z", 
            "score", "movement_vector_x", "movement_vector_y"])
            

    def write_frame_entry(self, frame_id, class_id, track_id, velocity, coordinate_x_opt, coordinate_y_opt, coordinate_world_x, coordinate_world_y, bottom_plate_1, bottom_plate_2, bottom_plate_3, bottom_plate_4, score, movement_vector):
        with open(self.file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([frame_id, class_id, track_id, velocity, 
            coordinate_x_opt, coordinate_y_opt, 
            coordinate_world_x, coordinate_world_y, 
            bottom_plate_1[0], bottom_plate_1[1], bottom_plate_1[2], 
            bottom_plate_2[0], bottom_plate_2[1], bottom_plate_2[2], 
            bottom_plate_3[0], bottom_plate_3[1], bottom_plate_3[2], 
            bottom_plate_4[0], bottom_plate_4[1], bottom_plate_4[2], 
            score, 
            movement_vector[0], movement_vector[1]])