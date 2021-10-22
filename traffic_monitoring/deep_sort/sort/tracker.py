# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import

import numpy as np

from . import iou_matching
from . import kalman_filter
from . import linear_assignment
from .kalman_filter import chi2inv95
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.8, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self._lambda = 0.7 # 0: only Appearance 1: only position

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _full_cost_metric(self, tracks, dets, track_indices, detection_indices):
        """
        This solves the missing Equation 5 from the DeepSort Paper
        Issue is opened here https://github.com/nwojke/deep_sort/issues/112
        Thanks to https://github.com/michael-camilleri/deep_sort/tree/302675fb15ba23c422fd70b4c6d72f65192889c3
        :param tracks:
        :type tracks:
        :param dets:
        :type dets:
        :param track_indices:
        :type track_indices:
        :param detection_indices:
        :type detection_indices:
        :return:
        :rtype:
        """
        pos_cost = np.empty([len(track_indices), len(detection_indices)])
        measurements = np.asarray([dets[i].to_xyah() for i in detection_indices])
        gating_threshold = chi2inv95[4]
        for row, track_idx in enumerate(track_indices):
            gating_distance = self.kf.gating_distance(tracks[track_idx].mean, tracks[track_idx].covariance, measurements, False)
            gating_distance = np.sqrt(gating_distance) / np.sqrt(gating_threshold)
            pos_cost[row, :] = gating_distance

        pos_gate = pos_cost > 1

        # Appearance based Metric
        app_cost = self.metric.distance(
            np.array([dets[i].feature for i in detection_indices]),
            np.array([tracks[i].track_id for i in track_indices]),
        )
        app_gate = app_cost > self.metric.matching_threshold

        app_cost[app_gate] = linear_assignment.INFTY_COST
        pos_cost[pos_gate] = linear_assignment.INFTY_COST


        cost_matrix = self._lambda * pos_cost + (1.0 - self._lambda) * app_cost
        cost_matrix[np.logical_or(pos_gate, app_gate)] = linear_assignment.INFTY_COST
        return cost_matrix

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)
            #cost_matrix[:, :] = linear_assignment.INFTY_COST - 1
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        # TODO if car then not doing this here

        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                self._full_cost_metric, linear_assignment.INFTY_COST -1, self.max_age,
               self.tracks, detections, confirmed_tracks)
        #unmatched_tracks_a = self.tracks
        #unmatched_detections = detections


        #track_indices = list(range(len(confirmed_tracks)))

        #detection_indices = list(range(len(detections)))

        #unmatched_tracks_a = list(set(track_indices))
        #unmatched_detections = detection_indices

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)
        matches = matches_a + matches_b
        #  matches = matches_b
        unmatched_tracks = list(set(unmatched_tracks_b + unmatched_tracks_a))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, segmentation):
        mean, covariance = self.kf.initiate(segmentation.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age, segmentation.generic_mask, segmentation.class_id, segmentation.score,
            segmentation.feature))
        self._next_id += 1
