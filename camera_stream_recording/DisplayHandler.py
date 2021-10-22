#  ****************************************************************************
#  @DisplayHandler.py
#
#
#
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

import cv2


class Display:
    """Class insert Text on a defined position on a frame with opencv. Just call the method 'overlay_text'"""
    def __init__(self, text_input, frame, pos):
        self.thickness = 1
        self.pos = pos
        self.frame = frame
        self.frame_out = None
        self.fontScale = 0.9
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.color = (255, 255, 255)
        self.lineType = cv2.LINE_AA
        self.text_input = str(text_input)

    def overlay_text(self):
        """Method uses opencv function putText to insert a text on a frame"""
        self.text_input = self.text_input
        self.frame_out = cv2.putText(self.frame, self.text_input, self.pos,
                                     self.font, self.fontScale, self.color,
                                     self.thickness, self.lineType)

        return self.frame_out
