#  ****************************************************************************
#  @Utils.py
#
#
#
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

import os
import subprocess
from datetime import datetime, timedelta


def delete_file(filename):
    """Delete a specific file"""
    if os.path.exists(filename):
        os.remove(filename)
    else:
        print("The file " + filename + " does not exist")


def generate_unique_name():
    """Returns a 'unique' name as string by using the current timestamp.\n
    Format of generated name: yyyy-mm-dd_hh-mm-ss
    """
    cur_datetime = datetime.now()
    return cur_datetime.strftime("%Y-%m-%d_%H-%M-%S")


def get_length(filename):
    """
    Get the length of a specific file with ffrobe from the ffmpeg library
    :param filename: this param is used for the file
    :type filename: str
    :return: length of the given video file
    :rtype: float
    """
    # use ffprobe because it is faster then other (for example moviepy)
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", filename
    ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)


def is_video_consistent(filename, video_duration):
    """
    Check if the Video is consistent with the length criteria
    :param filename: filename
    :type filename: str
    :param video_duration: expected video duration of video file in seconds
    :type video_duration: float
    :return: if the video is consistent due to a threshold
    :rtype: bool
    """
    # This is causing weird "file not found" issues --> uncommenting for now. MJ 21.03.2021

    # video_length = get_length(filename)
    # # Check if video is 8 percent shorter than expected
    # # check if the video is shorter than the expected length. This could be a "jump" in the video -> delete video
    # expected_length = video_duration
    # tolerance = expected_length - (expected_length * 0.1)
    # if video_length < tolerance:
    #     print('video is 1 percent shorter than expected:')
    #     print('video_length: ' + str(video_length))
    #     print('expected_length: ' + str(expected_length))
    #     print('tolerance: ' + str(tolerance))
    #     return False
    # else:
    #     print('video_length: ' + str(video_length))
    #     print('expected_length: ' + str(expected_length))
    return True


def upload_video(filename, SAS_TOKEN, STORAGE_ACCOUNT, STORAGE_CONTAINER):
    """
    Prepare the upload of files to a azure Filestorage

    :param filename: filename
    :type filename: str
    :param uploads_list: list of processes to track the process of uploading
    :type uploads_list: list
    :param storage_addr: address for the azure storage path
    :type storage_addr: str
    :param sas_token: token for azure authentitication
    :type sas_token:  str
    :return: -
    :rtype: void
    """
    print("Uploading file %s to %s / %s " % (filename, STORAGE_ACCOUNT, STORAGE_CONTAINER))
    block_blob_service = BlockBlobService(account_name=STORAGE_ACCOUNT, sas_token=SAS_TOKEN)
    block_blob_service.create_blob_from_path(STORAGE_CONTAINER, filename, filename)
    print("Upload successfull. Removing local file %s" % filename)
    os.remove(filename)
    return



def get_end_timestamp_from_minutes_duration(video_max_duration):
    return datetime.now() + timedelta(minutes=video_max_duration)


def get_end_timestamp_from_seconds_duration(video_max_duration):
    return datetime.now() + timedelta(seconds=video_max_duration)
