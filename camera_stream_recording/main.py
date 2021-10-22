#  ****************************************************************************
#  @main.py
#
#  Main Module for recording a rtsp video stream
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

import time

import cv2
import datetime
import subprocess
import tkinter as tk
from tkinter import simpledialog
import dateutil.parser
from decouple import config
import argparse
from pathlib import Path

from Utils import delete_file, is_video_consistent, get_end_timestamp_from_minutes_duration, \
    upload_video, generate_unique_name
from FrameDropHandler import FrameDropWatcher
from DisplayHandler import Display

DEFAULT_RUNTIME_M = 1440  # Default runtime duration in minutes
DEFAULT_VIDEO_DUR_M = 1  # Default video duration in minutes
UPLOAD_CHECKER_INTERVAL_M = 5  # Default duration after which the upload checker checks for completed uploads
VIS_RESIZE_FACTOR = 1  # resize factor how the image should be resized in vizualisation with imshow(). It's not


class MainCaptureHandling:
    """Main Module for recording a rtsp-stream in different modes"""

    def __init__(self, username, password, ip_addr, is_uploaded):
        self.fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.link = f"rtsp://{username}:{password}@{ip_addr}:80/axis-media/media.amp?streamprofile=rtspStreamLow"
        self.cap = cv2.VideoCapture(self.link)
        print(self.link)
        self.uploads = []
        self.is_uploaded = is_uploaded
        if not self.cap.isOpened():
            raise RuntimeError("Connection to camera is not working")

    def start_capture_one_video(self):
        """Initialize the Video Writer"""
        vid_name = generate_unique_name() + ".mp4"
        out = cv2.VideoWriter(vid_name, self.fourcc, 30.0,
                              (1920, 1080))
        return vid_name, out

    def handle_video_saving(self, filename, sub_procs_list, video_duration):
        """Method will start uploading videos and delete the files if there are not consistent"""
        if is_video_consistent(filename, video_duration):
            if self.is_uploaded:
                upload_video(filename, sub_procs_list, config('STORAGE_ADDR'), config('SAS_TOKEN'))
            else:
                print("No upload, because the script runs locally\n")
        else:
            print("The file will be deleted " + str(filename))
            delete_file(filename)

    def handle_video_saving_specific_time(self, filename, sub_procs_list,
                                          video_duration):
        """Method will start uploading and keep the files if there are not consistent"""
        if is_video_consistent(filename, video_duration):
            if self.is_uploaded:
                upload_video(filename, sub_procs_list, config('STORAGE_ADDR'), config('SAS_TOKEN'))
            else:
                print("No upload, because the script runs locally\n")
        else:
            print(
                "Video has some missing frames. I don't delete the file because we still need this "
                + filename)

    def check_for_completed_uploads(self, uploads_list, wait_time):
        """Method will execute the prepaired uploads and wait until the files are uploaded. If the file was uploaded
        with success, it will be deleted """
        rm_list = []
        for upload in self.uploads:
            try:
                exit_code = upload[0].wait(timeout=wait_time)

                if not (exit_code == 0):
                    print("Upload of file '" + upload[1] + "' failed.")
                else:
                    print("Delete file " + upload[1] +
                          "locally because it is on azure available")
                    delete_file(upload[1])
                    rm_list.append(upload)

            except subprocess.TimeoutExpired:
                pass
        for item in rm_list:
            uploads_list.remove(item)

    def capture_video_stream_for_specific_time(self, end_timestamp=None):
        """
        Captures one video with max duration or stopped by user
        :param end_timestamp: end timestamp when the video capturing will be finished
        :type end_timestamp: timestamp
        :return: None
        :rtype: void
        """

        self.vid_name, self.out = self.start_capture_one_video()

        start_timestamp = datetime.datetime.now()
        start_time = time.time()
        if end_timestamp is not None:
            print("Capturing stream until " + str(end_timestamp) + " for " +
                  str(end_timestamp - start_timestamp))

        count = 1
        #  f = open("relative_timestamps_" + self.vid_name + ".txt", "w+")
        #  f.write("count frame_number pts absolute_timestamp " + self.vid_name +
        #          "\n")

        print("Start recording " + str(start_timestamp) +
              " Press STRG + C to stop recording")

        stream_frame = None
        video_stream_instance = None
        cam_connect = False
        reset = False
        time_max_rate = 60
        drop_frame_count = 0
        connect_count = 0

        cam_connect = self.cap.read()
        time.sleep(0.5)
        cv2.namedWindow("CameraView", cv2.WINDOW_NORMAL)
        while True:
            prev_stream_frame = stream_frame
            try:
                #  pts = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                #  frame_number = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

                if not cam_connect or reset:
                    print("Reconnect Stream")
                    #  video_stream_instance = CameraVideoStream(self.link)
                    #  video_stream_instance.start()
                    #  cam_connect = video_stream_instance.grabbed
                    cam_connect = self.cap.read()
                    time.sleep(0.5)

                fwd_instance = FrameDropWatcher(prev_stream_frame,
                                                drop_frame_count,
                                                connect_count,
                                                self.cap,
                                                reset,
                                                drop_frame_max=7)
                fwd_instance.start()
                reset, drop_frame_count, stream_frame, connect_count = fwd_instance.read(
                )

                vis_frame = stream_frame
                vis_frame = cv2.resize(vis_frame,
                                       (int(vis_frame.shape[1] * VIS_RESIZE_FACTOR),
                                        int(vis_frame.shape[0] * VIS_RESIZE_FACTOR)))
                vis_frame = Display("Framedrop: " + str(drop_frame_count), vis_frame,
                                    (5, 60)).overlay_text()
                vis_frame = Display("Connect: " + str(connect_count), vis_frame,
                                    (5, 80)).overlay_text()
                if cam_connect:
                    cv2.imshow("CameraView", vis_frame)
                    self.out.write(stream_frame)

                    if end_timestamp is not None:
                        if datetime.datetime.now() >= end_timestamp:
                            break
                    count += 1
                time.sleep(max(1 / time_max_rate - (time.time() - start_time), 0))
                pressed_key = cv2.waitKey(1)
                if pressed_key & 0xFF == ord('q'):
                    print("q pressed. Stop Capturing")
                    break
            except KeyboardInterrupt:
                print("keyboard interrupt. Stop Capturing")
                break

        expected_length = (datetime.datetime.now() - start_timestamp).total_seconds()
        print("Video should have " + str(expected_length) + " Seconds length")
        self.out.release()
        fwd_instance.stop()
        self.handle_video_saving_specific_time(self.vid_name, self.uploads,
                                               expected_length)
        self.check_for_completed_uploads(self.uploads, None)
        cv2.destroyAllWindows()
        # self.match_with_first_timestamp()

    def match_with_first_timestamp(self):
        """
        Experiment Method for matching timestamps manually by manual "OCR" of the first frame
        :return:
        :rtype:
        """
        application_window = tk.Tk()
        video_capture = cv2.VideoCapture(self.vid_name)
        video_capture.set(cv2.CAP_PROP_FPS, 1)
        ret, frame = video_capture.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
        answer = simpledialog.askstring(
            "Date first image",
            "What is the timestamp format 2008-09-03T20:56:35.450686",
            parent=application_window)
        date = dateutil.parser.isoparse(answer)
        f = open("relative_timestamps_" + self.vid_name + "absolut_time.txt",
                 "w+")
        f.write("frame_number pts absolute_timestamp\n")
        relative_file = open("relative_timestamps_" + self.vid_name + ".txt",
                             "r")
        for idx, line in enumerate(relative_file.readlines()):
            if idx > 0:
                elems = line.split()
                absolute_timestamp = datetime.timedelta(
                    milliseconds=float(elems[2])) + date
                f.write(elems[1] + ' ' + str(absolute_timestamp) + '\n')

    def capture_video_stream_random_snippets(self, video_duration,
                                             runtime_duration):
        """
        Captures some videos during a specific runtime randomly without specific order
        :param video_duration: Duration of one video in minutes
        :type video_duration: int
        :param runtime_duration: Duration of the hole runtime of this script
        :type runtime_duration: int
        :return: None
        :rtype: None
        """

        self.vid_name, self.out = self.start_capture_one_video()

        print("Capturing stream for " + str(runtime_duration) +
              " minutes.\nVideo length is limited to " + str(video_duration) +
              " minutes.")
        end_time_script = datetime.datetime.now() + datetime.timedelta(
            minutes=runtime_duration)
        end_time_video = datetime.datetime.now() + datetime.timedelta(
            minutes=video_duration)
        end_time_upload_checker = datetime.datetime.now() + datetime.timedelta(
            minutes=UPLOAD_CHECKER_INTERVAL_M)
        while True:
            ret, frame = self.cap.read()
            if ret:
                # write the flipped frame
                self.out.write(frame)
                if datetime.datetime.now() >= end_time_script:
                    break
                if datetime.datetime.now() >= end_time_video:
                    self.out.release()
                    self.handle_video_saving(self.vid_name, self.uploads,
                                             video_duration * 60)
                    print('Video: ' + self.vid_name +
                          ' done. Next one is starting...')

                    if datetime.datetime.now() >= end_time_upload_checker:
                        self.check_for_completed_uploads(self.uploads, 1)
                        end_time_upload_checker = datetime.datetime.now(
                        ) + datetime.timedelta(
                            minutes=UPLOAD_CHECKER_INTERVAL_M)
                    end_time_video = datetime.datetime.now() + datetime.timedelta(minutes=video_duration)
                    self.vid_name, self.out = self.start_capture_one_video()

            else:
                print(
                    'Could not read a frame. So i delete the current video and start the next one...'
                )
                self.out.release()
                delete_file(self.vid_name)
                # check if script is done to avoid endless loop
                if datetime.datetime.now() >= end_time_script:
                    break
                # init new capture
                end_time_video = datetime.datetime.now() + datetime.timedelta(minutes=video_duration)
                self.vid_name, self.out = self.start_capture_one_video()

        self.cap.release()
        self.out.release()
        self.handle_video_saving(self.vid_name, self.uploads, video_duration)
        self.check_for_completed_uploads(self.uploads, None)

    def capture_images_every_minute(self, runtime_duration):
        end_time_script = datetime.datetime.now() + datetime.timedelta(
            minutes=runtime_duration)
        last_fetched_time = None
        count = 0
        folder_path = "random_images" + generate_unique_name()
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        while True:
            ret, frame = self.cap.read()
            if last_fetched_time is None or datetime.datetime.now() - last_fetched_time > datetime.timedelta(minutes=1):
                print(f"store image frame {count}")
                cv2.imwrite(folder_path + '/' + "frame%d.png" % count, frame)
                count += 1
                last_fetched_time = datetime.datetime.now()
            if datetime.datetime.now() >= end_time_script:
                break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Capture video stream with different options')
    parser.add_argument('-r',
                        '--random_mode',
                        action="store_true",
                        help="store rd videos with vd duration")
    parser.add_argument('-i', '--random_image_mode', action="store_true", help="store every minute one image during "
                                                                               "runtime. The runtime can specifid with"
                                                                               " -rd")
    parser.add_argument(
        '-p',
        '--planed_mode',
        action="store_true",
        help='planned mode, capture one video with vd duration. This is the mode for easy capturing one Video')
    parser.add_argument(
        '-u',
        '--upload',
        action="store_true",
        help="run the script with automatic uploading to azure storage")
    parser.add_argument('-vd',
                        '--video-duration',
                        metavar='N',
                        help='duration of a video snippet in minutes',
                        type=int)
    parser.add_argument('-rd',
                        '--runtime-duration',
                        metavar='N',
                        help='number of captured video in random mode',
                        type=int)
    #  if len(sys.argv) < 2:
    #    parser.print_help()
    #    sys.exit(1)
    args = parser.parse_args()
    video_capturing = MainCaptureHandling(config('CAMERA_USER_NAME'),
                                          config('PASSWORD'), config('CAMERA_IP_ADDR'), args.upload)

    if args.random_mode:
        vd = DEFAULT_VIDEO_DUR_M
        rd = DEFAULT_RUNTIME_M
        if args.video_duration:
            vd = args.video_duration
        if args.runtime_duration:
            rd = args.runtime_duration
        if vd > rd:
            raise RuntimeError(
                "Video duration can not be longer than run duration.")
        video_capturing.capture_video_stream_random_snippets(vd, rd)
    elif args.planed_mode:
        # Run only one planned capturing of one video user defined
        if args.video_duration:
            vdm_duration = args.video_duration
            video_capturing.capture_video_stream_for_specific_time(
                get_end_timestamp_from_minutes_duration(vdm_duration))
        else:
            video_capturing.capture_video_stream_for_specific_time()

    elif args.random_image_mode:
        rd = DEFAULT_RUNTIME_M
        if args.runtime_duration:
            rd = args.runtime_duration
        video_capturing.capture_images_every_minute(rd)
    else:
        print("Nothing specific selected, run simple capturing mode. You can quit Capturing with q in opencv or strg-c")
        video_capturing.capture_video_stream_for_specific_time()
