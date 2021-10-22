# Traing with Validation loss
# Reference: https://medium.com/@apofeniaco/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
#

import os

from detectron2.data import build_detection_test_loader, DatasetMapper, build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter

from ml_flow_detectron2 import MLflowMetricLogger
from .LossEvalHook import LossEvalHook


class TrainerWithValLoss(DefaultTrainer):
    """
    Trainer with validation loss
    Thanks to https://medium.com/@apofeniaco/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
    """

    @classmethod
    def build_train_loader(cls, cfg):
        dataloader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True,
                                                                            augmentations=cfg.AUGMENTATIONS))
        return dataloader

    def build_writers(self):
        return [
            CommonMetricPrinter(self.max_iter),
            MLflowMetricLogger(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, ("bbox", "segm"), True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                mapper=DatasetMapper(self.cfg, is_train=True, augmentations=[])
            )
        ))
        return hooks
