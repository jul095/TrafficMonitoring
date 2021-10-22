#  ****************************************************************************
#  @evaluation_tracking.py
#
#  Evaluation of tracking methods SORT vs. Deep SORT
#
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

import os
from pathlib import Path

import cv2
import motmetrics as mm
import numpy as np

from TrackerSelector import TrackerSelector, TrackingMode
from geomapping.CameraCalibration import CameraCalibration
from position_estimation.EstimateVehicleBasePlate import EstimateVehicleBasePlate
from read_dataset.ReadCOCODataset import ReadCOCODatasets
from util import convert_XYWH_bbox_to_XCYCWH
from vis.Visualizer import Visualizer

OUTPUT_FOLDER = 'eval_tracking_output'


def xyxy_to_tlwh(bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy

    t = x1
    l = y1
    w = int(x2 - x1)
    h = int(y2 - y1)
    return t, l, w, h


class TrackingEvaluator:
    """
    Evaluate the Tracking algorithm
    to compare SORT vs. Deep SORT Tracking and evaluate both with motmetrics
    """

    def __init__(self):
        """
        Choose the tracking algorithm and select the challenge like percentage frame drop
        In this case Ground Truth labels will be dropped randomly to challenge the tracking
        """
        self.acc = mm.MOTAccumulator(auto_id=True)
        self.accs = []
        self.visualizer = Visualizer()
        self.trackingMode = TrackingMode.SORT
        self.is_frame_ignore = True
        self.frame_drop_percentage = 0.2
        self.tracker = TrackerSelector(self.trackingMode)

    def reset_accumulator(self):
        """
        accumulator counts all the metrics together
        """
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, trk_tlwhs, trk_ids, gt_trk_tlwhs, gt_trk_ids):
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)
        iou_distance = mm.distances.iou_matrix(gt_trk_tlwhs, trk_tlwhs, max_iou=0.5)
        return self.acc.update(gt_trk_ids, trk_ids, iou_distance)

    def run_eval_for_all_datasets(self, read_dataset):
        """
        run the tracking algorithm for all datasets and accumulate the tracking metrics
        """
        self.accs = []
        for dataset in read_dataset:
            print(dataset[0].get('file_name'))
            print(self.get_gt_tack_count(dataset))
            acc = self.run_eval_for_one_dataset(dataset, is_frame_ignore=self.is_frame_ignore,
                                                frame_drop_percentage=self.frame_drop_percentage)
            self.accs.append(acc)
            self.get_summary()
            self.reset_accumulator()
        self.get_summary_all_datasets()

    def get_gt_tack_count(self, dataset):
        all_tracks_count = []
        for data in dataset:
            gt_tracks_id = []
            for elem in data.get("annotations"):
                gt_track_id = elem.get("track_id")
                gt_tracks_id.append(gt_track_id)
            all_tracks_count.extend(gt_tracks_id)
        return len(set(all_tracks_count))

    def run_eval_for_one_dataset(self, dataset, is_frame_ignore, frame_drop_percentage):
        """
        evaluate tracking of one dataset with SORT or Deep SORT
        """
        estimate_3d_bbox = EstimateVehicleBasePlate()
        classes = ['car', 'cyclist', 'car trailer', 'truck', 'truck trailer', 'car-transporter', 'motorcycle', 'bus',
                   'police car', 'firefighter truck', 'ambulance', 'pedestrian', 'predestrian with stroller',
                   'pedestrian in wheelchair', 'scooter', 'transporter']
        camera = CameraCalibration()

        ignore_frames_id = []
        step_for_step = 1
        if is_frame_ignore:
            all_frame_count = len(dataset)
            if Path(f'ignore_frame_ids_{all_frame_count}_{frame_drop_percentage}.txt').is_file():
                # pass
                ignore_frames_id = np.loadtxt(
                    os.path.join(OUTPUT_FOLDER, 'ignore_frame_ids_{all_frame_count}_{frame_drop_percentage}.txt'))
            else:
                ignore_frames_id = np.random.randint(0, all_frame_count, int(all_frame_count * frame_drop_percentage),
                                                     dtype=np.int).astype(dtype=np.int)
                np.savetxt(
                    os.path.join(OUTPUT_FOLDER, f'ignore_frame_ids_{all_frame_count}_{frame_drop_percentage}.txt'),
                    ignore_frames_id, newline=" ")

        for frame_id, data in enumerate(dataset):
            """per Frame"""
            frame = cv2.imread(data["file_name"])

            gt_bboxes = []
            gt_classes = []
            gt_masks = []
            gt_scores = []
            gt_tracks_id = []
            gt_trk_tlwhs = []

            if frame_id not in ignore_frames_id:

                for elem_count, elem in enumerate(data.get("annotations")):
                    """per Label in one Frame"""
                    gt_class_id = int(elem.get('category_id'))
                    gt_track_id = elem.get('track_id')
                    bbox = elem.get('bbox')
                    polygon_in_one_row = np.asarray(elem.get('segmentation'))[0]
                    polygon = polygon_in_one_row.reshape(-1, 2)
                    label_text = "Class ID: %i Track ID: %i" % (gt_class_id, gt_track_id)

                    # frame = self.visualizer.draw_mask_with_polygon(frame, polygon_in_one_row, bbox, gt_class_id, label_text)

                    #  bit_mask = convert_polygon_to_mask(polygon)
                    #  gen_mask = GenericMask(bit_mask, 1080, 1920)
                    gt_masks.append(polygon_in_one_row)
                    gt_classes.append(gt_class_id)
                    gt_bboxes.append(convert_XYWH_bbox_to_XCYCWH(bbox))
                    gt_trk_tlwhs.append(bbox)
                    gt_tracks_id.append(gt_track_id)
                    gt_scores.append(0.99)
            else:
                step_for_step = 1
                print("frame %d skipped " % frame_id)
            tracking_result = self.tracker.update_tracking_with_labeled_data(frame, gt_bboxes, gt_masks, gt_scores,
                                                                             gt_classes)

            trk_tlwhs, track_ids = [], []

            for bb in tracking_result:
                bbox_xyxy = bb['bbox']
                track_id = bb['track_id']
                trk_tlwhs.append(xyxy_to_tlwh(bbox_xyxy))
                track_ids.append(track_id)

                # converted_polygon = bb['generic_mask'].astype(int).reshape(-1,2)
                # print(converted_polygon)
                # middle_point, bottom_plate = estimate_3d_bbox.find_rotated_bbox_and_get_middlepoint_world(frame,
                #                                                                                          converted_polygon,
                #                                                                                          track_id,
                #                                                                                          classes[
                #                                                                                              bb['class_id']])
                # pixel_middle_point = camera.projection_world_to_pixel(middle_point.copy())

                # cv2.circle(frame, (int(pixel_middle_point[0]), int(pixel_middle_point[1])), 2, (0, 0, 0), 2)

                frame = self.visualizer.draw_bbox_xyxy(frame, bb['bbox'], int(bb['track_id']))
                # frame = self.visualizer.draw_mask_with_polygon(frame, bb['generic_mask'], bb['bbox'], bb['class_id'],
                #                                              "Track ID: " + str(bb['track_id']))

            self.eval_frame(trk_tlwhs, track_ids, gt_trk_tlwhs, gt_tracks_id)

            cv2.imshow("evaluation tracking", frame)
            cv2.waitKey(step_for_step)

        return self.acc

    def get_summary(self):
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=['num_frames', 'mota', 'motp', 'num_objects', 'num_unique_objects',
                                                'num_predictions',
                                                'num_fragmentations', 'mostly_lost', 'mostly_tracked', 'num_switches'])
        self.print_summary(mh, summary)

    def get_summary_all_datasets(self):
        mh = mm.metrics.create()
        summary = mh.compute_many(self.accs, metrics=['num_frames', 'mota', 'motp', 'num_objects', 'num_unique_objects',
                                                      'num_predictions',
                                                      'num_fragmentations', 'mostly_lost', 'mostly_tracked',
                                                      'num_switches'],
                                  generate_overall=True)
        self.print_summary(mh, summary)

    def print_summary(self, mh, summary):
        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        print(strsummary)

        summary_file = open(
            os.path.join(OUTPUT_FOLDER,
                         f'{"SORT" if self.trackingMode is TrackingMode.SORT else "DEEPSORT"}_{self.is_frame_ignore}_{self.frame_drop_percentage if self.is_frame_ignore else ""}.txt'),
            'w')
        summary_file.write(strsummary)
        summary_file.close()


if __name__ == '__main__':
    read_dataset_test = ReadCOCODatasets("training").get_dataset_seperated()
    read_dataset_test.extend(ReadCOCODatasets("validation").get_dataset_seperated())
    read_dataset_test.extend(ReadCOCODatasets("test").get_dataset_seperated())
    tracking_evaluator = TrackingEvaluator()
    tracking_evaluator.run_eval_for_all_datasets(read_dataset_test)
    # tracking_evaluator.get_summary_all_datasets()
