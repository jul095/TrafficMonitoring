#  ****************************************************************************
#  @Visualizer.py
#
#  Visualizer class for drawing the results with OpenCV
#
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

import random

import cv2
import numpy as np

# Some nice visual colors for segmentation or bboxes
COLORS = ((20, 111, 236), (21, 89, 42), (255, 138, 138), (75, 93, 209), (196, 84, 183), (36, 166, 64), (33, 144, 166),
          (171, 34, 171), (190, 204, 41), (70, 219, 152), (181, 104, 217), (214, 150, 47), (68, 179, 108),
          (255, 235, 59), (255, 193, 7), (255, 152, 0), (255, 235, 59), (255, 193, 7), (255, 152, 0), (255, 235, 59),
          (255, 193, 7), (255, 152, 0))

# Initial color mapping
# 0: bicycle, 1: car, 2: person, 3: transporter
COLOR_MAPPING = {0: (196, 84, 183), 1: (33, 144, 166), 2: (181, 104, 217), 3: (20, 111, 236)}


class Visualizer:
    """
    Visualizer class for visualize Segmentations, bounding boxes and text with category depended random color
    """

    def __init__(self):
        self.color_mapping = COLOR_MAPPING

    def get_color_mapping(self, label_id):
        """
        generates a random color mapping each category get it's own color if no initial color mapping is there
        """
        if label_id in self.color_mapping:
            return self.color_mapping[label_id]
        color = random.choice(COLORS)
        color = (int(color[0]), int(color[1]), int(color[2]))
        self.color_mapping[label_id] = color
        return color

    def get_darker_color_mapping(self, label_id, decrement=70):
        """
        decrements the color for better visualization and returns a rgb code
        """
        r, g, b = self.get_color_mapping(label_id)
        r -= decrement if r > decrement else 0
        g -= decrement if g > decrement else 0
        b -= decrement if b > decrement else 0
        return r, g, b

    def draw_bbox_xyxy(self, frame, bbox, track_id):
        """
        Draw Bounding Box in xyxy format and return a painted frame
        """
        x1, y1, x2, y2 = [int(i) for i in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 50, 0), 2)
        cv2.putText(frame, str("id") + str(track_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        return frame

    def get_center_of_bitmask(self, bitmask):
        _num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(bitmask, 8)
        largest_component_id = np.argmax(stats[1:, -1]) + 1
        return np.median((cc_labels == largest_component_id).nonzero(), axis=1)[::-1]

    def _calculate_optimal_text_pos_with_bitmask(self, int_mask, label_width):
        """
        returns optimal pixel position for the label box depending on the bitmask segmentation
        """
        #  in stats[1] there is the first element the x component left up, second y left up, third x right up ...
        _num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(int_mask, 8)
        c1, c2 = (int(stats[1][0]), int(stats[1][1])), (stats[1][0] + label_width, stats[1][1] - 20)
        return c1, c2

    def draw_textbox(self, frame, bitmask_or_bbox, category_id, label_text, line_thickness=2):
        """
        create textbox with rectanglge box and optimized position and returns annotated frame
        """
        tl = line_thickness or round(0.002 * (frame.shape[0] + frame.shape[1]) / 2) + 1

        font_scale = tl / 3
        font = cv2.QT_FONT_NORMAL
        thickness = 1
        # Text Size for estimate rectangle box
        (label_width, label_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        #  Get dimension of text label for the right size of the Rectangle
        if hasattr(bitmask_or_bbox, 'shape'):
            c1, c2 = self._calculate_optimal_text_pos_with_bitmask(bitmask_or_bbox, label_width)
        else:
            c1, c2 = (int(bitmask_or_bbox[0]), int(bitmask_or_bbox[1])), (
                int(bitmask_or_bbox[0] + label_width), int(bitmask_or_bbox[1] - 20))

        cv2.rectangle(frame, c1, c2, self.get_color_mapping(category_id), -1, cv2.LINE_AA)
        cv2.putText(frame, label_text, (c1[0], c1[1] - 2), font, font_scale, [0, 0, 0], thickness=thickness,
                    lineType=cv2.LINE_AA)
        return frame

    def _draw_polygon_and_overlay(self, frame, polygon, category_id, alpha=0.3):
        """
        return weighted overlay of a polygon with category specific color
        """
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], self.get_color_mapping(category_id))
        cv2.polylines(overlay, [polygon], True, self.get_darker_color_mapping(category_id), thickness=1,
                      lineType=cv2.LINE_AA)
        return cv2.addWeighted(frame, alpha, overlay, 1 - alpha, 0.0)

    def draw_mask_with_polygon(self, frame, polygon, bbox, category_id, label_text, alpha=0.3):
        """
        Draw a segmentation mask but with a polygon inserted. This is good for working with ground truth data from cvat
        :param frame: image frame
        :type frame: frame
        :param polygon: polygon of the segmentation
        :type polygon: 2d list
        :param bbox: bounding box of segmentation for finding the optimal place for the label text
        :type bbox: list
        :param category_id: id of detected object type to add a correct color
        :type category_id: int
        :param label_text: text for showing information above the segmentation
        :type label_text: str
        :param alpha: transparency value
        :type alpha: float
        :return: frame
        :rtype: frame
        """
        converted_polygon = polygon.astype(int).reshape(-1, 2)
        frame = self.draw_textbox(frame, bbox, category_id, label_text)
        return self._draw_polygon_and_overlay(frame, converted_polygon, category_id, alpha)

    def draw_mask_with_mask(self, frame, masks, category_id, label_text, alpha=0.3):
        """
        Draw a segmentation mask with text label on a frame
        This method is good for real data from the detectron2 mask rcnn network
        :param frame: Frame
        :type frame: frame
        :param masks: Generic Mask object from detectron2
        :type masks: GenericMask
        :param category_id: id of detected object type to add a correct color
        :type category_id: int
        :param label_text: text for showing information above the segmentation
        :type label_text: str
        :param alpha: transparency value
        :type alpha: float
        :return: frame
        :rtype: frame
        """
        converted_mask = masks.mask.astype("uint8")
        converted_polygon = masks.polygons[0].astype(int).reshape(-1, 2)
        #  center = get_center_of_bitmask(converted_mask)
        frame = self.draw_textbox(frame, converted_mask, category_id, label_text)
        return self._draw_polygon_and_overlay(frame, converted_polygon, category_id, alpha)

    def draw_rotated_bbox(self, frame, rectangle, color=(0, 0, 255), thickness=2):
        return cv2.drawContours(frame, [rectangle], 0, color, thickness)

    def plot_trajectories(self, points, frame, category_id=None, line_thickness=None):
        tl = line_thickness or round(0.002 * (frame.shape[0] + frame.shape[1]) / 2) + 1  # line/font thickness
        color = self.get_color_mapping(category_id) or [np.random.randint(0, 255) for _ in range(3)]
        return cv2.polylines(frame, points, False, color, thickness=tl, lineType=cv2.LINE_AA)
