#  ****************************************************************************
#  @train.py
#
#  Training script to train own Mask R-CNN model
#
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

import argparse
import os
from datetime import datetime
from random import random

import cv2
import detectron2.data.transforms as T
import mlflow
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.model_zoo import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import GenericMask
from detectron2.utils.visualizer import Visualizer as DetectronVisualizer
from loss_eval_hook import TrainerWithValLoss
from read_dataset import ReadCOCODatasets, KITTIFile, COCOFile
from vis import Visualizer

"""
Usage:
- detectron2 https://github.com/facebookresearch/detectron2 (Apache-2.0 License)
"""


# important for correct logging of detectron2
setup_logger()


def register_datasets(dir_path, dataset_type):
    """
    Register coco-Dataset with method from detectron2
    """
    registered_dataset_ids = []
    dataset_dir = [
        name for name in os.listdir(dir_path)
        if os.path.isdir(os.path.join(dir_path, name))
    ]

    for dataset in dataset_dir:
        dataset_id = dataset_type + "_" + os.path.basename(dataset)
        json_loc = os.path.join(dir_path, dataset, "annotations",
                                "instances_default.json")
        img_loc = os.path.join(dir_path, dataset, "images")

        register_coco_instances(dataset_id, {}, json_loc, img_loc)

        registered_dataset_ids.append(dataset_id)
    return registered_dataset_ids


def short_demo(cfg, dataset_test):
    """
    Test and visualize inference on a test dataset
    """
    predictor = DefaultPredictor(cfg)
    own_vis = Visualizer()
    for d in dataset_test:
        img = cv2.imread(d['file_name'])
        visualizer = DetectronVisualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("training_kitti"), scale=1)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("ground trough", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)

    #    for object in d:
    #        mask = object['categories']
    #        segmentation = object['segmentation']

    #

    for d in dataset_test:
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        outputs = outputs["instances"]
        predictions = outputs.pred_boxes.tensor.cpu().numpy()
        classes = outputs.pred_classes.cpu().numpy()
        _masks = outputs.pred_masks.cpu()
        scores = outputs.scores.cpu().numpy()

        for _class, mask, score in zip(classes, _masks, scores):
            # x0, y0, x1, y1 = box
            # bbox_xcycwh.append([(x1 + x0) / 2, (y1 + y0) / 2, (x1 - x0), (y1 - y0)])
            generic_mask = GenericMask(np.asarray(mask), 375, 1242)
            own_vis.draw_mask_with_mask(img, generic_mask, _class, f"Class ID: {_class}")

        # out = visualizer.draw_instance_predictions(output["instances"].to("cpu"))
        cv2.imshow("prediction", img)
        cv2.waitKey(0)


def prepare_config(train_dataset_ids, val_dataset_ids, category_count, args):
    """
    Configure hyper parameters for Mask RCNN in Detectron2
    """
    cfg = get_cfg()
    cfg.merge_from_file('./maskrcnn/mask_rcnn_R_50_FPN_3x.yaml')
    #  Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    #  Dataset Config
    # Training Dataset
    cfg.DATASETS.TRAIN = train_dataset_ids
    # Validation Dataset (Not Test!)
    cfg.DATASETS.TEST = val_dataset_ids

    # cfg.INPUT.MIN_SIZE_TRAIN = (1080,)
    # cfg.INPUT.MAX_SIZE_TRAIN = (1920,)
    # cfg.INPUT.MIN_SIZE_TEST = (1080,)
    # cfg.INPUT.MAX_SIZE_TEST = (1920,)

    cfg.DATALOADER.NUM_WORKERS = 1

    #  Hyper params
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.WARMUP_ITERS = args.warmup_iters
    cfg.SOLVER.GAMMA = args.gamma
    cfg.SOLVER.STEPS = args.steps
    cfg.SOLVER.CHECKPOINT_PERIOD = args.save_interval

    # Model config

    # It's more performance necessary if unfreeze this with 0, so the default is 2
    cfg.MODEL.BACKBONE.FREEZE_AT = args.freeze_at

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

    # cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    # cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_BBOX_REG = True

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = category_count

    # Validation config
    cfg.TEST.EVAL_PERIOD = args.validation_interval  # Validation Period

    #  Training without Augmentations
    # cfg.AUGMENTATIONS = []

    #  Default Augmentations in Detectron2
    cfg.AUGMENTATIONS = [
        T.ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1920, sample_style='choice'),
        T.RandomFlip()]

    # cfg.AUGMENTATIONS =  [T.RandomBrightness(0.8, 1.8),
    #                      T.RandomContrast(0.6, 1.3),
    #                      T.RandomSaturation(0.8, 1.4),
    #                      T.RandomFlip(prob=1),
    #                      T.RandomLighting(0.7), ]

    #  output folder
    cfg.OUTPUT_DIR = f"./output/{args.result_subfolder}"

    #  create output folder if not exist
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def parse_args():
    """
    Configure default hyperparams and cli interface
    """
    parser = argparse.ArgumentParser("Training script for Traffic Monitoring")

    parser.add_argument('--batch_size', default=1, type=int,
                        help='Model batch size')
    parser.add_argument('--lr', default=0.00025, type=float,
                        help='Model learning rate')
    parser.add_argument('--max_iter', default=80000, type=int,
                        help='Maximum Iterations')
    parser.add_argument('--validation_interval', default=1000, type=int,
                        help="Validation uses validation Dataset for validation")
    parser.add_argument('--gamma', default=0.1, type=float,
                        help="Weight decay")
    parser.add_argument('--steps', default=(30000,), type=int,
                        help="Learning rate steps. Pass argument as '--steps x y'  without '='",
                        nargs='+')
    parser.add_argument('--save_interval', default=5000, type=int,
                        help="Saves a model checkpoint every n steps")
    parser.add_argument('--warmup_iters', default=1000, type=int,
                        help="learning rate warmup steps")
    parser.add_argument('--freeze_at', default=2, type=int,
                        help="Freezes the Network at block x -> 0 to train all layers")
    parser.add_argument('--output_dir', default="./model_output/", type=str,
                        help="Output dir for the checkpoints and log files")
    parser.add_argument('--eval_only', default=False, type=bool, help="only perform evaluation of model")
    parser.add_argument('--eval_only_pretraining', default=False, type=bool,
                        help="only perform evaluation of pretraining")

    now = datetime.now
    parser.add_argument('--result_subfolder', default=now().strftime('%Y-%m-%d-%H%M%S'), type=str,
                        help="subfolder name")

    return parser.parse_args()


def main(trainer):
    """
    Main training loop
    """
    if args.eval_only:
        #  only perform evaluation of validation dataset

        model = TrainerWithValLoss.build_model(cfg)
        # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)
        # res = TrainerWithValLoss.test(cfg, model)
        # if comm.is_main_process():
        #     verify_results(cfg, res)

        evaluator = COCOEvaluator("test", ("bbox", "segm"), True, output_dir="./output/")
        val_loader = build_detection_test_loader(cfg, "test")

        metrics = inference_on_dataset(model, val_loader, evaluator)
        return metrics
    return trainer.train()


def evaluate_only_pretraining():
    """
    evaluate only the pretraining of detectron2 model zoo with own dataset and return metrics
    """
    cfg = get_cfg()
    cfg.merge_from_file('./maskrcnn/mask_rcnn_R_50_FPN_3x.yaml')
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    read_dataset_training = ReadCOCODatasets("validation", is_coco_eval=True)
    DatasetCatalog.register('test_chaoskreuzung', lambda: read_dataset_training.get_detectron2_dataset())

    things_classes = MetadataCatalog.get("coco_2017_train").thing_classes

    MetadataCatalog.get('test_chaoskreuzung').set(thing_classes=things_classes)

    cfg.DATASETS.TEST = ("test_chaoskreuzung",)
    cfg.freeze()

    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("test_chaoskreuzung", ("bbox", "segm"), False, output_dir="./output/")
    test_loader = build_detection_test_loader(cfg, "test_chaoskreuzung")
    metrics = inference_on_dataset(predictor.model, test_loader, evaluator)
    print(metrics)
    return metrics


def evaluate(cfg, test_dataset="test"):
    """
    Test final model with the test dataset and return metrics
    """
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,
                                     "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
    cfg.DATASETS.TEST = test_dataset
    predictor = DefaultPredictor(cfg)
    # model = build_model(cfg)
    # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #    cfg.MODEL.WEIGHTS, resume=True
    # )
    evaluator = COCOEvaluator(test_dataset, ("bbox", "segm"), True, output_dir=os.path.join(cfg.OUTPUT_DIR))
    val_loader = build_detection_test_loader(cfg, test_dataset)
    metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(metrics)
    return metrics


def log_mlflow_hyperparams(cfg):
    """
    Log chosen hyperparameters to mlflow
    """
    mlflow.log_param("max_iterations", cfg.SOLVER.MAX_ITER)
    mlflow.log_param("batch_size_per_image_backbone", cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE)
    mlflow.log_param("num_classes", cfg.MODEL.ROI_HEADS.NUM_CLASSES)

    mlflow.log_param("base_lr", cfg.SOLVER.BASE_LR)
    mlflow.log_param("freeze_at", cfg.MODEL.BACKBONE.FREEZE_AT)
    mlflow.log_param("batch_size", cfg.SOLVER.IMS_PER_BATCH)

    mlflow.log_param("warmup_iters", cfg.SOLVER.WARMUP_ITERS)
    mlflow.log_param("gamma", cfg.SOLVER.GAMMA)
    mlflow.log_param("steps", cfg.SOLVER.STEPS)

    mlflow.log_param("datasets_train", cfg.DATASETS.TRAIN)
    mlflow.log_param("datasets_validation", cfg.DATASETS.TEST)

    mlflow.log_param("data_augmentations", cfg.AUGMENTATIONS)

    mlflow.log_param("eval_period", cfg.TEST.EVAL_PERIOD)


def log_mlflow(cfg, metrics):
    """
    Log training configuration and test results to mlflow
    """
    # https://towardsdatascience.com/object-detection-in-6-steps-using-detectron2-705b92575578
    # https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=h9tECBQCvMv3

    for metricType in metrics:
        for metricName in metrics[metricType]:
            logged_name = metricName + "_" + metricType
            mlflow.log_metric(logged_name.lower(), metrics[metricType][metricName])
            print(metricType + metricName + str(metrics[metricType][metricName]))

    print("Start logging metrics to mlflow...")
    # mlflow.pytorch.log_model(model,
    #                         os.path.join(cfg.OUTPUT_DIR,
    #                                      "model_final.pth"))  # Save the model in a format to be registered by MLflow
    mlflow.log_artifact(os.path.join(cfg.OUTPUT_DIR, "metrics.json"))  # Save the metrics as json to the storage
    mlflow.log_artifact(
        os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))  # Save the model in detectron2 format to the blob storage
    print("DONE")


if __name__ == '__main__':
    args = parse_args()
    if args.eval_only_pretraining:
        evaluate_only_pretraining()
        exit()

    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data/train2017/')

    #  init Dataset input reader

    # COCO 2017 Dataset
    read_coco_train_2017 = COCOFile(dataset_path, "instances_train2017")
    #  own labeled training data
    read_dataset_training = ReadCOCODatasets("training")
    # own labeled validation data
    read_dataset_validation = ReadCOCODatasets("validation")
    # own labeled test data
    read_dataset_test = ReadCOCODatasets("test")
    # kitti dataset
    read_dataset_training_kitti = KITTIFile()


    #  paste the dataset reader in a dataset catalog for detectron2
    DatasetCatalog.register('training_coco', lambda: read_coco_train_2017.get_detectron2_dataset_original_coco())
    DatasetCatalog.register('training', lambda: read_dataset_training.get_detectron2_dataset())
    DatasetCatalog.register('training_kitti', lambda: read_dataset_training_kitti.get_detectron2_dataset())
    DatasetCatalog.register('validation', lambda: read_dataset_validation.get_detectron2_dataset())
    DatasetCatalog.register('test', lambda: read_dataset_test.get_detectron2_dataset())

    #  Get the category names in correct order
    things_classes = read_dataset_training._coco_files[0].get_detectron2_metadata()
    #  Count the length to choose the size of the output layer
    category_count = len(things_classes)

    MetadataCatalog.get('training_coco').set(thing_classes=things_classes)
    MetadataCatalog.get('training').set(thing_classes=things_classes)
    MetadataCatalog.get('training_kitti').set(thing_classes=things_classes)
    MetadataCatalog.get('validation').set(
        thing_classes=things_classes)
    MetadataCatalog.get('test').set(
        thing_classes=things_classes)

    count_gpus = torch.cuda.device_count()
    print(f"Number of GPUs in system: {count_gpus}")

    cfg = prepare_config(('training',), ('validation',), category_count, args)
    #  short_demo(cfg, read_dataset_training_kitti.get_detectron2_dataset())

    #  Possible remote tracking for mlflow
    #  remote_server_uri = "http://chaosflow.westeurope.cloudapp.azure.com:5000"
    #  mlflow.set_tracking_uri(remote_server_uri)  # set MLflow tracking server
    mlflow.set_experiment('master_thesis_optimized_training')
    trainer = TrainerWithValLoss(cfg)
    #  metrics = evaluate(cfg, trainer)

    trainer.resume_or_load(resume=False)
    log_mlflow_hyperparams(cfg)
    main(trainer)

    #  For multi gpu training
    #  launch(main, count_gpus, num_machines=1, args=(cfg,), )

    metrics = evaluate(cfg)
    log_mlflow(cfg, metrics)
