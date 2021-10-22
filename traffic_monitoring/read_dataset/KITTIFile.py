#  ****************************************************************************
#  @KITTIFile.py
#
#  KITTI Dataset Reader
#
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

import glob
import os

import cv2
import numpy as np
import pycocotools.mask as rletools
from detectron2.structures import BoxMode
from imantics import Mask


class SegmentedObject:
    """
    Segmentation class for loading KITTI dataset into the detectron2 format
    """

    def __init__(self, mask, class_id, track_id):
        self.mask = mask
        self.class_id = class_id
        self.track_id = track_id


class KITTIFile:
    """
    Class for loading only the KITTI-Dataset into detectron2
    """

    def __init__(self):
        self.label_path = os.path.join(os.path.dirname(__file__), '../..',
                                       'data/kitti_datasets/rtwh_labels/')
        self.image_path = os.path.join(os.path.dirname(__file__), '../..', 'data/kitti_datasets/training/image_02/')

    def load_sequences(self, path, seqmap):
        objects_per_frame_per_sequence = {}
        for seq in seqmap:
            print("Loading sequence", seq)
            seq_path_folder = os.path.join(path, seq)
            seq_path_txt = os.path.join(path, seq + ".txt")
            if os.path.isdir(seq_path_folder):
                objects_per_frame_per_sequence[seq] = self.get_detectron2_dataset(seq_path_folder)
            elif os.path.exists(seq_path_txt):
                objects_per_frame_per_sequence[seq] = self.load_label_txt_file(seq_path_txt)
            else:
                assert False, "Can't find data in directory " + path

        return objects_per_frame_per_sequence

    def load_txt(self, path):
        objects_per_frame = {}
        track_ids_per_frame = {}  # To check that no frame contains two objects with same id
        combined_mask_per_frame = {}  # To check that no frame contains overlapping masks
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                fields = line.split(" ")

                frame = int(fields[0])
                if frame not in objects_per_frame:
                    objects_per_frame[frame] = []
                if frame not in track_ids_per_frame:
                    track_ids_per_frame[frame] = set()
                if int(fields[1]) in track_ids_per_frame[frame]:
                    assert False, "Multiple objects with track id " + fields[1] + " in frame " + fields[0]
                else:
                    track_ids_per_frame[frame].add(int(fields[1]))

                class_id = int(fields[2])
                if not (class_id == 1 or class_id == 2 or class_id == 10):
                    assert False, "Unknown object class " + fields[2]

                mask = {'size': [int(fields[3]), int(fields[4])], 'counts': fields[5].encode(encoding='UTF-8')}
                if frame not in combined_mask_per_frame:
                    combined_mask_per_frame[frame] = mask
                elif rletools.area(rletools.merge([combined_mask_per_frame[frame], mask], intersect=True)) > 0.0:
                    assert False, "Objects with overlapping masks in frame " + fields[0]
                else:
                    combined_mask_per_frame[frame] = rletools.merge([combined_mask_per_frame[frame], mask],
                                                                    intersect=False)
                objects_per_frame[frame].append(SegmentedObject(
                    mask,
                    class_id,
                    int(fields[1])
                ))

        return objects_per_frame

    def get_detectron2_dataset(self):
        """
        returns only images with persons on the image in the necessary detectron2 format
        """
        image_folders = sorted(glob.glob(os.path.join(self.image_path, "*")))
        image_folders = image_folders[0:2]
        dataset_dicts = []
        person_counter = 0
        for image_folder in image_folders:
            objects_per_frame = self.load_txt(os.path.join(self.label_path, f"{os.path.basename(image_folder)}.txt"))
            images_per_folder = sorted(glob.glob(os.path.join(image_folder, "*.png")))

            for frame_id, image in enumerate(images_per_folder):
                assert frame_id == int(os.path.basename(image).split('.')[0])
                if frame_id in objects_per_frame.keys():
                    objects_in_current_frame = objects_per_frame[frame_id]
                    record = {}
                    record["file_name"] = image
                    record[
                        "image_id"] = f"{os.path.basename(image)}{os.path.basename(image_folder)}"  # Take care if this number is unique!!!
                    record["height"] = objects_in_current_frame[0].mask['size'][0]
                    record["width"] = objects_in_current_frame[0].mask['size'][1]
                    objs = []
                    if 2 in [ann.class_id for ann in objects_in_current_frame]:
                        #  2 is pedestrian, 1 car in https://www.vision.rwth-aachen.de/page/mots
                        person_counter += 1
                        for ann in objects_in_current_frame:
                            bitmask = rletools.decode(ann.mask)
                            polygons = Mask(bitmask).polygons()
                            polygon_points = polygons.points[0].reshape((-1))
                            # if len(polygon_points) >= 6 and len(polygon_points) % 2 == 0:
                            obj = {}
                            obj["segmentation"] = [polygon_points]
                            obj["category_id"] = 2 if ann.class_id == 2 else 1
                            polygons = np.float32(polygons.points[0])

                            x, y, w, h = cv2.boundingRect(polygons)
                            bbox = [x, y, w, h]
                            obj["bbox"] = bbox
                            obj["bbox_mode"] = BoxMode.XYWH_ABS
                            objs.append(obj)
                            # else:
                            #    pass
                            # print("pass", ann.class_id)

                    if len(objs) > 0:
                        record["annotations"] = objs
                        dataset_dicts.append(record)

        print(person_counter)
        return dataset_dicts

    def filename_to_frame_nr(self, filename):
        assert len(filename) == 10, "Expect filenames to have format 000000.png, 000001.png, ..."
        return int(filename.split('.')[0])
