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


class ExtractTrajectories:
    """
    Store the Trajectories in a csv file
    """
    def __init__(self, file_name):
        self.file_name = file_name

        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["frame_id", "class_id", "track_id", "velocity", "coordinate_world_x_opt", "coordinate_world_y_opt", "coordinate_world_x", "coordinate_world_y"])

    def write_frame_entry(self, frame_id, class_id, track_id, velocity, coordinate_x_opt, coordinate_y_opt, coordinate_world_x, coordinate_world_y):
        with open(self.file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([frame_id, class_id, track_id, velocity, coordinate_x_opt, coordinate_y_opt, coordinate_world_x, coordinate_world_y])
