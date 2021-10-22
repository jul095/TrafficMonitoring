#  ****************************************************************************
#  @TrackerSelector.py
#
#  Adapted:
#  - deep_sort_pytorch https://github.com/ZQPei/deep_sort_pytorch (MIT License)
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

import numpy as np
from detectron2.utils.visualizer import GenericMask

from deep_sort import DeepSort
from util import convert_XCYCWH_bbox_to_XYXY


class TrackingMode:
    SORT = 1
    DEEP_SORT = 2


class TrackerSelector:
    """
    Meta class to select the tracking algorithm of your choise
    """

    def __init__(self, tracking_mode=TrackingMode.DEEP_SORT):
        '''

        :param tracking_mode: sort or deep_sort
        :type tracking_mode: int
        '''
        self.tracking_mode = tracking_mode

        if self.tracking_mode == TrackingMode.SORT:
            pass
            #  removed due to license issues
            #  self.sort_tracker = Sort()
        elif self.tracking_mode == TrackingMode.DEEP_SORT:
            self.deep_sort_tracker = DeepSort("deep_sort/deep/checkpoint/ckpt.t7", use_cuda=True)
        else:
            raise Exception('Tracking mode not available')

    def update_tracking_with_labeled_data(self, image, bboxes, masks, scores, cls_ids):
        if self.tracking_mode == TrackingMode.SORT:
            final_list = []
            for bbox, score in zip(bboxes, scores):
                bbox_xyxy = convert_XCYCWH_bbox_to_XYXY(bbox)
                final_list.append([bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]])
            if len(final_list) == 0:
                return self.sort_tracker.update()
            else:
                return self.sort_tracker.update(np.array(final_list, dtype=np.float64))
        return self.deep_sort_tracker.update(np.array(bboxes, dtype=np.float64), masks,
                                             np.array(scores, dtype=np.float64), np.array(cls_ids, dtype=np.uint8),
                                             image)

    def update(self, output, image):
        if self.tracking_mode == TrackingMode.SORT:
            pass
            # remove due to license issues
            return self.sort_tracker.update(output)
        elif self.tracking_mode == TrackingMode.DEEP_SORT:

            predictions = output.pred_boxes.tensor.cpu().numpy()
            classes = output.pred_classes.cpu().numpy()
            _masks = output.pred_masks.cpu()
            scores = output.scores.cpu().numpy()

            bbox_xcycwh, masks, cls_conf, cls_ids = [], [], [], []
            for box, _class, mask, score in zip(predictions, classes, _masks, scores):
                x0, y0, x1, y1 = box
                bbox_xcycwh.append([(x1 + x0) / 2, (y1 + y0) / 2, (x1 - x0), (y1 - y0)])
                masks.append(GenericMask(np.asarray(mask), 1080, 1920))
                cls_conf.append(score)
                cls_ids.append(_class)

            return self.deep_sort_tracker.update(np.array(bbox_xcycwh, dtype=np.float64), masks,
                                                 np.array(cls_conf, dtype=np.float64),
                                                 np.array(cls_ids, dtype=np.uint8), image)
        else:
            pass
