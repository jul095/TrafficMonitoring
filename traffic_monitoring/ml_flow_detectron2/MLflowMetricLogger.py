#  ****************************************************************************
#  @MLflowMetricLogger.py
#
#  Logging metrics to MLFlow
#
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

import mlflow
import torch
from detectron2.utils.events import EventWriter, get_event_storage


class MLflowMetricLogger(EventWriter):
    """
    Override base class EventWriter to add mlflow metrics
    """
    def __init__(self, max_iter):
        """
        Args:
            max_iter (int): the maximum number of iterations to train.
                Used to compute ETA.
        """
        self._max_iter = max_iter
        self._last_write = None

    def write(self):
        storage = get_event_storage()
        iteration = storage.iter
        if iteration == self._max_iter:
            # This hook only reports training progress (loss, ETA, etc) but not other data,
            # therefore do not write anything after training succeeds, even if this method
            # is called.
            return

        try:
            date_time = storage.history("date_time").avg(20)
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            date_time = -1
        try:
            lr = storage.history("lr").latest()
        except KeyError:
            lr = -1
        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = -1

        losses = [[k, v.median(20)] for k, v in storage.histories().items() if "loss" in k]
        mlflow.log_metric("iteration", iteration)

        for loss in losses:
            mlflow.log_metric(loss[0], loss[1], step=iteration)
        mlflow.log_metric("date_time", float(date_time), step=iteration)
        mlflow.log_metric("learning_rate", lr, step=iteration)
        mlflow.log_metric("memory", max_mem_mb, step=iteration)
