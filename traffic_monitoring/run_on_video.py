#  ****************************************************************************
#  @run_on_video.py
#
#  Usage:
#  - detectron2 https://github.com/facebookresearch/detectron2 (Apache-2.0 License)
#
#  Adapted:
#  - deep_sort_pytorch https://github.com/ZQPei/deep_sort_pytorch (MIT License)
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

import argparse
from datetime import datetime
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from shapely.geometry import Polygon
from tqdm import tqdm

from TrackerSelector import TrackerSelector, TrackingMode
from extract_trajectories import ExtractTrajectories
from geomapping import CameraCalibration
from position_estimation import TrackedObject, EstimateVehicleBasePlate
import azure_upload_file
from util import calc_center
from vis import Visualizer
from MetadataOutput import MetadataOutput

"""
Main Python Module for traffic trajectory extraction
"""


def parse_args():
    parser = argparse.ArgumentParser('Extract trajectories based on video')
    parser.add_argument('--video', default=None, type=str, help="specific mp4 video file")
    parser.add_argument('--passpoints', default='./config/frame0_measurement.png.points', type=str, help="specific passpoint matching file for georeferencing")
    parser.add_argument('--valid-area', default='./config/valid_area.csv', type=str, help="specific area definition. Within this pixel area segmentation are evaluated")
    return parser.parse_args()


def prepare_config(category_count):
    """
    Configure Detectron2, trained model and Mask R-CNN hyperparameter
    """
    cfg = get_cfg()
    cfg.merge_from_file('./maskrcnn/mask_rcnn_R_50_FPN_3x.yaml')
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    #    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = category_count
    cfg.OUTPUT_DIR = './model_weights'
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,
                                     "model_final.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'training_0421', "model_0004999.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50  # set a custom testing threshold if a segmentation is shown
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.99
    return cfg


def extract_video_file_timestamp(video_file):
    return video_file[video_file.rindex(os.sep) + 1:video_file.rindex('.')]


def prepare_video_processing(video_file, video_source_folder, cfg):
    """
    Initialize opencv methods videoCapture and videoWriter
    """
    video = cv2.VideoCapture(video_file)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_writer = cv2.VideoWriter(
        os.path.join(video_source_folder, 'videos_output', f"output{extract_video_file_timestamp(video_file)}.mp4"),
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=float(frames_per_second), frameSize=(width, height), isColor=True)
    predictor = DefaultPredictor(cfg)
    return video, predictor, video_writer, num_frames


def prepare_valid_area(frame, valid_area_file_path):
    """
    Reads a csv file in pixel coordinates for a geofence at the image borders
    """
    valid_area_df = pd.read_csv(valid_area_file_path)
    valid_area = valid_area_df.to_numpy()
    valid_area = valid_area.reshape((-1, 1, 2))
    #cv2.polylines(frame, [valid_area], True, (255, 0, 0), thickness=2)


def check_if_in_valid_area(polygon_segmentation):
    """
    Checks if an object is in the area
    """
    valid_area_df = pd.read_csv('config/valid_area.csv')
    valid_area = Polygon(valid_area_df.to_numpy())
    polygon = Polygon(polygon_segmentation)
    return polygon.within(valid_area)


def run_on_video(video_capture, predictor, max_frames, video_file_timestamp, video_file_folder, passpoints_file_path, valid_area_file_path, category_names=None):
    """
    Main Method for handling traffic trajectory extraction
    """
    currentTime = datetime.now()
    read_frames = 0
    tracker = TrackerSelector(TrackingMode.DEEP_SORT)
    estimate_base_plate = EstimateVehicleBasePlate(passpoints_file_path)
    csvPath = os.path.join(video_file_folder, 'trajectory_output', f"{video_file_timestamp}_trajectories.csv")
    extract_trajectories = ExtractTrajectories(csvPath)

    vis = Visualizer()
    tracked_object_dict = {}
    trajectory_points_dict = {}
    camera = CameraCalibration(passpoints_file_path)

    #Read parameters for transforming world coordinates into pixel coordinates
    world_file = np.loadtxt(os.path.join('config', 'crossing_map_referenced_with_camera_image.pgw')).reshape((3, 2))
    offset_X_world, offset_Y_world, camera_matrix, rotation_matrix, translation_vector = camera.return_world_to_pixel_parameters()

    # get frames and duration
    fps = video_capture.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = -1
    try:
        duration = frame_count/fps
    except Exception:
        print("Division error! (frame_count/fps) ", frame_count, "/", fps)


    #Write Json with Video Metadata
    jsonPath = video_file_folder + '/videos_output/' + video_file_timestamp + '.json'

    metadataOutput = MetadataOutput(jsonPath)
    metadataOutput.extendOutput({"timestamp": video_file_timestamp, "csvPath": csvPath,"frame_count": frame_count, "duration": duration, "offset_X_world": offset_X_world, "offset_Y_world": offset_Y_world, "camera_matrix": camera_matrix.tolist(), "rotation_matrix": rotation_matrix.tolist(), "translation_vector": translation_vector.tolist()})
    highestTrackId = 0

    while True:

        is_frame, frame = video_capture.read()

        if read_frames < 0:
            continue
        if not is_frame:
            break

        outputs = predictor(frame)
        predictions = outputs["instances"]
        track_bbs_ids = tracker.update(predictions, frame)

        for bb in track_bbs_ids:

            track_id = bb['track_id']
            category_id = bb['class_id']

            bbox = bb['bbox']
            score = bb['score'] * 100

            middle_point_bbox = calc_center(bbox)
            prepare_valid_area(frame, valid_area_file_path)

            trajectory_points_dict[track_id] = trajectory_points_dict.get(track_id, [])
            trajectory_points_dict[track_id].append(middle_point_bbox)

            converted_polygon = bb['generic_mask'].polygons[0].astype(int).reshape(-1, 2)

            # Tracked Object for estimate movement direction for localize the reference point at
            # middle of the rear axle
            tracked_object_dict[track_id] = tracked_object_dict.get(track_id, TrackedObject(track_id, category_id))
            tracked_object: TrackedObject = tracked_object_dict[track_id]

            if check_if_in_valid_area(converted_polygon):

                world_center_point_bbox = camera.projection_pixel_to_world(
                    np.array([middle_point_bbox[0], middle_point_bbox[1], 1]).reshape(3, 1))

                bottom_plate = [[NaN, NaN, NaN], [NaN, NaN, NaN], [NaN, NaN, NaN], [NaN, NaN, NaN]]

                if category_names[category_id] in ['truck', 'car', 'transporter']:
                    middle_point, bottom_plate = estimate_base_plate.find_base_plate_and_center_world(frame,
                                                                                                      converted_polygon,
                                                                                                      category_names[
                                                                                                          category_id])
                    optimized_center_point, velocity = tracked_object.add_bottom_plate_and_center(bottom_plate,
                                                                                                  middle_point,
                                                                                                  world_center_point_bbox)
                else:
                    optimized_center_point = world_center_point_bbox
                    velocity = tracked_object.add_bbox_and_center(bbox, world_center_point_bbox)

                pixel_middle_point = camera.projection_world_to_pixel(optimized_center_point.copy())

                movement_vector = tracked_object.movement_vector if tracked_object.movement_vector is not None else [NaN, NaN]
                extract_trajectories.write_frame_entry(read_frames, category_names[category_id], track_id, velocity,
                                                       optimized_center_point[0],
                                                       optimized_center_point[1], world_center_point_bbox[0],
                                                       world_center_point_bbox[1], bottom_plate[0],
                                                       bottom_plate[1], bottom_plate[2], bottom_plate[3],
                                                       score, movement_vector)

                if len(trajectory_points_dict[track_id]) > 1:
                    coords_array = np.array([trajectory_points_dict[track_id]], dtype=np.int32)
                    #  frame = vis.plot_trajectories(coords_array, frame, category_id, line_thickness=3, )

                if category_names is not None:
                    label_text = "%10.2f %s %i Vel: %i km/h" % (score, category_names[category_id], track_id, velocity)
                else:
                    label_text = "Class ID: %i Track ID: %i Vel: %i km/h" % (category_id, track_id, velocity)

                #  frame = vis.draw_bbox_xyxy(frame,bbox,track_id)
                frame = vis.draw_mask_with_mask(frame, bb['generic_mask'], category_id, label_text)
                cv2.circle(frame, (int(pixel_middle_point[0]), int(pixel_middle_point[1])), 2, (255, 255, 0), 2)
                #  cv2.circle(frame, middle_point_bbox, 2, (0, 0, 255), 2)
                if highestTrackId < track_id : highestTrackId = track_id
        #cv2.rectangle(frame, (0, 0), (565, 36), (166,166,166), -1) # hide timestamp
        #cv2.imshow("frame", frame)
        #cv2.waitKey(1)

        #  visualization = visualizer.draw_instance_predictions(frame, outputs["instances"].to("cpu"))
        #  visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)
        yield frame
        read_frames += 1
        if read_frames > max_frames:
            break
    metadataOutput.extendOutput({"numberOfTrackIds": highestTrackId})
    metadataOutput.writeOutput()

    



def process_video(video_file, categories, passpoints_file_path, valid_area_file_path):
    """
    main entry point which runs the trajectory extraction in a loop
    """

    video_file_timestamp =  extract_video_file_timestamp(video_file)
    video_source_folder = os.path.dirname(video_file)

    #Write Json with Video Metadata
    jsonPath = video_source_folder + '/videos_output/' + video_file_timestamp + '.json'
    currentTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadataOutput = MetadataOutput(jsonPath)
    metadataOutput.extendOutput({"Status": "Processing", "startTime": currentTime})

    cfg = prepare_config(len(categories))

    Path(os.path.join(video_source_folder, 'videos_output')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(video_source_folder, 'trajectory_output')).mkdir(parents=True, exist_ok=True)

    video_capture, predictor, video_writer, max_frames = prepare_video_processing(video_file, video_source_folder, cfg)
    for visualization in tqdm(
            run_on_video(video_capture, predictor, max_frames, video_file_timestamp,
                         video_source_folder, passpoints_file_path, valid_area_file_path, categories),
            total=max_frames):
        video_writer.write(visualization)
    video_capture.release()
    video_writer.release()
    
    currentTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadataOutput.extendOutput({"endTime": currentTime, "Status": "Processed", "videoPath": video_file})
    metadataOutput.writeOutput()

def postProcessing(video_file):
    #print ("video_file", video_file)
    video_path = os.path.dirname(video_file)
    #print ("video_path", video_path)

    video_name = os.getenv('VIDEO_NAME', 'video.mp4')
    #print ("video_name", video_name)
    video_name = video_name.split(".")[0]
    #print ("video_name", video_name)

    azure_upload_file.upload_file(os.path.join(video_path, "trajectory_output"), video_name + "_trajectories.csv")
    azure_upload_file.upload_file(os.path.join(video_path, "videos_output"), video_name + ".json")

if __name__ == '__main__':

    #  change the categories here otherwise a error will appear if the output size of Mask R-CNN is not suitable
    #  classes = ['ambulance', 'bicycle', 'bus', 'car', 'person', 'scooter', 'transporter', 'truck']
    # categories = ['car', 'cyclist', 'car trailer', 'truck', 'truck trailer', 'car-transporter', 'motorcycle',
    #                   'bus', 'police car', 'firefighter truck', 'ambulance', 'pedestrian', 'predestrian with stroller',
    #                   'pedestrian in wheelchair', 'scooter', 'transporter']
    categories = ['bicycle', 'car', 'person', 'transporter']

    args = parse_args()

    video_file = args.video
    if args.video:
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        passpoints_file_path = os.path.abspath(args.passpoints)
        process_video(args.video, categories, args.passpoints, args.valid_area)
        postProcessing(args.video)