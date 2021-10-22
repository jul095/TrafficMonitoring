#  ****************************************************************************
#  @FrameDropHandler.py
#
#
#
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

from threading import Thread, Lock
import numpy as np


class FrameDropWatcher:
    """Class checks, if a frame will be dropped by comparing each frame with another. If a threshold of frames are
     dropped we estimate the connection is lost"""
    def __init__(self,
                 prev_frame=None,
                 drop_frame_count=None,
                 connect_count=None,
                 capture=None,
                 reset_stat=None,
                 drop_frame_max=None):
        """
        Constructor for initializing FrameDropWatcher
        :param prev_frame: Frame stored before
        :param drop_frame_count: current count of dropped frames
        :param connect_count: current count of connections lost
        :param capture: opencv VideoCapture instance
        :param reset_stat:
        :param drop_frame_max: threshold if we detect a lost connection
        """
        self.thread = Thread(target=self.update)
        self.thread.daemon = False
        self.started = False
        self.read_lock = Lock()

        _, self.currentframe = capture.read()
        self.prev_frame = prev_frame
        self.drop_frame_count = drop_frame_count
        self.capture = capture
        self.reset_stat = reset_stat
        self.connect_count = connect_count
        self.drop_frame_max = drop_frame_max

    def start(self):
        if self.started:
            print("already started!!")
            return None
        self.started = True
        self.thread.start()
        return self

    def update(self):
        """
        Reads frame and if frame is not received tries to reconnect the camera
        """
        if self.started:
            self.read_lock.acquire()

            if np.array_equal(self.prev_frame, self.currentframe):
                self.drop_frame_count += 1

                if self.drop_frame_count >= self.drop_frame_max:
                    # print("reConnect")
                    self.capture.stop()
                    self.reset_stat = True
                    self.connect_count += 1
                    self.drop_frame_count = 0
            else:
                self.drop_frame_count = 0
                self.reset_stat = False

            self.read_lock.release()

    def read(self):
        """
        :return: metrics and current frame
        :rtype: (reset_stat, drop_frame_count, currentframe, connect_count)
        """
        self.read_lock.acquire()
        self.currentframe = self.currentframe.copy()
        self.drop_frame_count = self.drop_frame_count
        self.reset_stat = self.reset_stat
        self.connect_count = self.connect_count
        self.read_lock.release()
        return self.reset_stat, self.drop_frame_count, self.currentframe, self.connect_count

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback):
        self.capture.release()
