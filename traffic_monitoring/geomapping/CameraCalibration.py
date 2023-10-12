#  ****************************************************************************
#  @CameraCalibration.py
#
#  Calibration method for the Camera
#  for the mapping between world and pixel coordinates
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

import os

import cv2
import numpy as np

from geomapping.GeoApplication import GeoApplication

#  Offset to increase the accuracy of Camera Calibration
#  because opencv only calculates with float32
OFFSET_X_WORLD = 678000
OFFSET_Y_WORLD = 5400000


class CameraCalibration:
    """
    Class for mapping between pixel and world (geographic) coordinate system
    """

    def __init__(self, passpoints_file_path):

        path = os.path.abspath(__file__)
        self.dir_path = os.path.dirname(path)

        self.geo_application = GeoApplication(passpoints_file_path)

        self.world_points = []
        self.pixel_points = []

        self.camera_matrix = None
        self.dist_coeffs = None
        self.rotation_matrix = None
        self.translation_vector = None

        self.load_and_prepare_pixel_and_world_points()
        self.calculate_intrinsic_and_extrinsic_parameter()

    def evaluate_error_reprojection(self, rotation_vector, translation_vector,
                                    camera_matrix, dist_coeffs):
        """
        Calculate the reprojection error of transforming the world point of the point pairs to a pixel
        point and compare the difference with the real pixel with the l2 norm
        The opencv-method projectPoints will be compared by the own implementation self.projection_world_to_pixel
        """
        mean_error_opencv_method = 0
        mean_error_own_implementation = 0

        for i in range(len(self.world_points)):
            img_points_project_points, _ = cv2.projectPoints(self.world_points[i], rotation_vector,
                                                             translation_vector,
                                                             camera_matrix, dist_coeffs)

            real_world_points = np.asarray(
                [self.world_points[i][0] + OFFSET_X_WORLD, self.world_points[i][1] + OFFSET_Y_WORLD, 0],
                dtype=np.float64)

            img_points_own_implementation = self.projection_world_to_pixel(real_world_points)

            #  Norm L2 is the euclidean distance (square root of sum of squares)
            error_own_implementation = cv2.norm(np.asarray([[self.pixel_points[i]]]),
                                                np.asarray([[img_points_own_implementation]]), cv2.NORM_L2) / len(
                img_points_own_implementation)
            error = cv2.norm(np.asarray([[self.pixel_points[i]]]), np.float32(img_points_project_points),
                             cv2.NORM_L2) / len(
                img_points_project_points)

            # x, y, = self.geo_application.convert_pixel_to_world(self.geo_application.H, self.pixel_points[i][0],
            #                                                   self.pixel_points[i][1])
            # print("HOMO", x, y)
            # print(img_points_own_implementation)
            # print(self.world_points[i].astype(np.double) * 100.)
            mean_error_opencv_method += error
            mean_error_own_implementation += error_own_implementation

        print("total error reprojection: {} pixels".format(mean_error_opencv_method / len(self.world_points)))
        print("total error own reprojection: {} pixels".format(mean_error_own_implementation / len(self.world_points)))

    def evaluate_error_projection(self):
        mean_error = 0
        for i in range(len(self.world_points)):
            pixels_xy = self.pixel_points[i]
            pixels = np.append(pixels_xy, 1)
            pixels.shape = (3, 1)
            world_points = self.projection_pixel_to_world(pixels)
            error = np.linalg.norm(np.subtract(world_points, self.world_points[i]))
            mean_error += error
        print("total error projection: {}".format(mean_error / len(self.world_points)))

    def load_and_prepare_pixel_and_world_points(self):
        """
        Load point pairs and subtract a offset for numberic accuracy
        """
        pixel_points, world_points = self.geo_application.calibrate_camera_with_given_points_by_qgis()
        world_points = [[point[0] - OFFSET_X_WORLD, point[1] - OFFSET_Y_WORLD, 0.0] for point in world_points]
        world_points = np.asarray(world_points)
        self.world_points = np.asarray(np.around(world_points, 2), dtype=np.float32)
        self.pixel_points = np.asarray(np.around(pixel_points, 2), dtype=np.float32)

    def calculate_intrinsic_and_extrinsic_parameter(self):
        """
        Calculate Translation, Rotation (extrinsic) and camera matrix (intrinsic)
        """
        frame = cv2.imread(os.path.join(self.dir_path, '../config/frame0_measurement.png'))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, camera_matrix, dist_coeffs, rotation_vector, translation_vector = cv2.calibrateCamera([self.world_points],
                                                                                                 [self.pixel_points],
                                                                                                 gray.shape[::-1],
                                                                                                 None, None)

        im_size = (frame.shape[1], frame.shape[0])

        self.camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, im_size, 0,
                                                                im_size)

        #  dist_coeffs is set to zero because the camera has a activated barrel distortion correction (bdc)
        dist_coeffs = np.zeros((8, 1))

        ret, new_rotation_vector, self.translation_vector = cv2.solvePnP(
            np.asarray(self.world_points, dtype=np.float32), self.pixel_points,
            self.camera_matrix,
            dist_coeffs, tvec=translation_vector[0], rvec=rotation_vector[0], useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE)

        self.rotation_matrix = cv2.Rodrigues(new_rotation_vector)[0]

        self.evaluate_error_reprojection(self.rotation_matrix, self.translation_vector, self.camera_matrix,
                                         dist_coeffs)

    def projection_pixel_to_world(self, pixel_point):
        """

        :param pixel_point:
        :type pixel_point:
        :return:
        :rtype:
        """
        position_inv_rotation_cam_pixel = np.linalg.inv(self.rotation_matrix).dot(
            np.linalg.inv(self.camera_matrix).dot(pixel_point))

        inv_r_and_translation = 0 + self.rotation_matrix.transpose().dot(self.translation_vector)[2, 0]

        s = inv_r_and_translation / position_inv_rotation_cam_pixel[2, 0]

        position = s * position_inv_rotation_cam_pixel
        full_position = position - self.rotation_matrix.transpose().dot(self.translation_vector)

        return np.asarray(
            [full_position[0, 0] + OFFSET_X_WORLD, full_position[1, 0] + OFFSET_Y_WORLD, full_position[2, 0]])

    def projection_world_to_pixel(self, world_position):
        world_position.shape = (3, 1)
        world_position[0] = world_position[0] - OFFSET_X_WORLD
        world_position[1] = world_position[1] - OFFSET_Y_WORLD
        pixels_s = self.camera_matrix.dot(self.rotation_matrix.dot(world_position) + self.translation_vector)
        pixel = pixels_s / pixels_s[2]
        pixel = np.around(pixel, 2).astype(dtype=np.float32)
        return np.asarray([pixel[0][0], pixel[1][0]], dtype=np.float32)

    def return_world_to_pixel_parameters(self):
        return OFFSET_X_WORLD, OFFSET_Y_WORLD, self.camera_matrix, self.rotation_matrix, self.translation_vector


if __name__ == '__main__':
    camera = CameraCalibration()
