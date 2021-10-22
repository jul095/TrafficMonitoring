#  ****************************************************************************
#  @GeoApplication.py
#
#  Calibration method for the Camera
#  for the mapping between world and pixel coordinates
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

import csv
import os
import tkinter as tk
from tkinter import *

import cv2
import matplotlib.image as mpimg
import numpy as np
from PIL import ImageTk, Image


class GeoApplication:

    def __init__(self):
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        self.points_file_path = os.path.join(dir_path, '../config/frame0_measurement.png.points')
        self.image_path = '../config/frame0_measurement.png'
        self.points_image = []
        self.points_citygml = []
        self.start_with_qgis()

    def start_with_qgis(self):
        self.points_image, self.points_citygml = self.calibrate_camera_with_given_points_by_qgis()
        self.H, _ = cv2.findHomography(np.float32(self.points_image), np.float32(self.points_citygml))

    def start_from_scratch(self):
        self.choose_image_points_in_opencv()
        self.input_real_world_gcp()
        self.H, _ = cv2.findHomography(np.float32(self.points_image), np.float32(self.points_citygml))

    def click_and_add(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.calibration_image, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
            cv2.putText(self.calibration_image, 'ID: ' + str(len(self.points_image)), (x + 5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2, cv2.LINE_AA)
            print('Image point clicked: ', x, y)
            self.points_image.append([x, y])

    def click_and_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x_cood, y_cood = self.convert_pixel_to_world(self.H, x, y)
            print('Image point: ', x, y, 'Real World', x_cood, y_cood)
            cv2.circle(self.img, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
            cv2.putText(self.img, 'GPSG_32632: ' + str(x_cood) + ', ' + str(y_cood), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

    def convert_pixel_to_world(self, H, x, y):
        pt1 = np.array([x, y, 1])
        pt1 = pt1.reshape(3, 1)
        pt2 = np.dot(H, pt1)
        pt2 = pt2 / pt2[2]
        return pt2[0], pt2[1]

    def calibrate_camera_with_given_points_by_qgis(self):
        points_image = []
        points_citygml = []
        with open(self.points_file_path) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for row in csv_reader:
                points_image.append([float(row['pixelX']), (-1.) * float(row['pixelY'])])
                points_citygml.append([float(row['mapX']), float(row['mapY'])])
        return points_image, points_citygml

    def choose_image_points_in_opencv(self):
        self.calibration_image = cv2.imread(self.image_path)
        cv2.namedWindow("configuremapping", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("configuremapping", self.click_and_add)
        while True:
            cv2.imshow("configuremapping", self.calibration_image)
            k = cv2.waitKey(100)
            if k == 27:
                print("ESC")
                cv2.destroyAllWindows()
                break
            if cv2.getWindowProperty("configuremapping", cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()

    def extract_gcp_values_from_ui(self):
        for ui_elem_x, ui_elem_y in self.list_ui_elems:
            self.points_citygml.append([float(ui_elem_x.get()), float(ui_elem_y.get())])
        self.root.destroy()

    def input_real_world_gcp(self):
        self.root = tk.Tk()

        self.frame = tk.Frame(self.root, width=100, height=100)
        self.frame.grid(row=0, column=0)
        self.lmain = tk.Label(self.frame)
        self.lmain.grid(row=0, column=0)
        self.calibration_image = cv2.imread(self.image_path)
        for idx, image_point in enumerate(self.points_image):
            cv2.circle(self.calibration_image, (image_point[0], image_point[1]), radius=2, color=(0, 0, 255),
                       thickness=-1)
            cv2.putText(self.calibration_image, 'ID: ' + str(idx), (image_point[0] + 5, image_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (200, 200, 0), 2, cv2.LINE_AA)

        cv2image = cv2.cvtColor(self.calibration_image, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        self.imgtk = ImageTk.PhotoImage(image=img)
        self.lmain.configure(image=self.imgtk)

        self.list_ui_elems = []

        for idx, image_point in enumerate(self.points_image):
            label = Label(master=self.root, bg='#FFCFC9', text='Point ID ' + str(idx))
            label.grid(row=idx, column=1)
            x_label = StringVar(self.root, value='x-value')
            y_label = StringVar(self.root, value='y-value')
            self.list_ui_elems.append((Entry(master=self.root, bg='white', textvariable=x_label),
                                       Entry(master=self.root, bg='white', textvariable=y_label)))
            x_elem, y_elem = self.list_ui_elems[idx]
            x_elem.grid(row=idx, column=2)
            y_elem.grid(row=idx, column=3)

        self.buttonok = Button(self.root, text='OK and calculate H Matrix', width=25,
                               command=self.extract_gcp_values_from_ui)
        self.buttonok.grid(row=len(self.points_image), column=1)
        self.root.mainloop()

    def show_click_demo(self):
        originimage = mpimg.imread(self.image_path)
        # img_warped = cv2.warpPerspective(originimage, H, (originimage.shape[1], originimage.shape[0]))
        self.img = cv2.imread(self.image_path)
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("frame", self.click_and_point)

        while True:
            cv2.imshow("frame", self.img)
            k = cv2.waitKey(100)
            if k == 27:
                print("ESC")
                cv2.destroyAllWindows()
                break
            if cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()

    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.root.destroy()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    geoApp = GeoApplication()
    #  geoApp.start_from_scratch()
    geoApp.start_with_qgis()
    geoApp.show_click_demo()
