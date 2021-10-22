#  ****************************************************************************
#  @ReadCOCODataset.py
#
#  Read COCO Datasets with filtering methods and different category matching
#
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

import os
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util
import pycocotools.mask as rletools
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from imantics import Mask
from pycocotools.coco import COCO


class ReadCOCODatasets:
    """
    Meta class for collecting different several instances of COCOFile
    """

    def __init__(self, data_type='training', vis=False, is_coco_eval=False):
        """
        Search datasets in the dataset folder and load them into a list
        """
        self.data_type = data_type
        self._coco_files = []
        self.is_coco_eval = is_coco_eval
        dataset_path = os.path.join(os.path.dirname(__file__), '../..', 'data/dataset',
                                    data_type)
        folder_of_datasets = os.listdir(dataset_path)
        print(dataset_path)
        for folder_dataset in folder_of_datasets:
            print(folder_dataset)
            coco_file = COCOFile(os.path.join(dataset_path, folder_dataset))
            self._coco_files.append(coco_file)
            self.categories = coco_file.categories
        if vis:
            #  self.plot_statistic_with_raw_categories()
            self.plot_statistics_with_merged_categories()
        self.get_unique_count_of_categories()

    def get_unique_count_of_categories(self):
        """
        Get some statistics about the track cound of the dataset
        """
        image_statistic = {}
        for coco_file in self._coco_files:
            processed_track_id = {-1}
            for existing_label in coco_file.coco.anns:
                annotation = coco_file.coco.loadAnns(existing_label)
                track_id = annotation[0]["attributes"]["track_id"]
                if track_id not in processed_track_id:
                    new_category, new_category_id = coco_file.get_mapped_category_by_old_id(
                        annotation[0]["category_id"])
                    if new_category_id is not None:
                        current_count = image_statistic[new_category_id] if new_category_id in image_statistic else 0
                        image_statistic[new_category_id] = current_count + 1
                        processed_track_id.add(track_id)
        return image_statistic

    def get_detectron2_dataset(self):
        """
        get the dataset prepared in the necessary detectron2 format
        """
        dataset_dicts = []
        for coco_file in self._coco_files:
            if self.data_type == "training":
                # If we want the training dataset, we only want relevant images
                # So we have to filter the dataset
                dataset_dict = coco_file.get_detectron2_dataset(True, coco_category_ids=self.is_coco_eval)
            else:
                dataset_dict = coco_file.get_detectron2_dataset(True, coco_category_ids=self.is_coco_eval)
            dataset_dicts.extend(dataset_dict)
        return dataset_dicts

    def get_dataset_seperated(self):
        dataset_list = []
        for coco_file in self._coco_files:
            dataset_list.append(coco_file.get_detectron2_dataset(True))
        return dataset_list

    def get_count_label_in_barchart(self, axes, data):
        """
        create barchart for data statistics
        """
        for idx, data_category_label_count in enumerate(data):
            if data_category_label_count == 0:
                axes.text(x=idx - 0.5, y=20, s=f"{data_category_label_count}",
                          fontdict=dict(fontsize=15), va='center')
            else:
                axes.text(x=idx - 0.5, y=0 + 100 / data_category_label_count, s=f"{data_category_label_count}",
                          fontdict=dict(fontsize=15), va='center')

    def autolabel(self, rects, axes):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            axes.text(rect.get_x() + rect.get_width() / 2., 0.99 * height,
                      '%d' % int(height),
                      ha='center', va='bottom', fontsize=14)

    def plot_statistics_with_merged_categories(self):
        """
        Barcharts for all COCOfile objects
        """
        dataset_dicts = self.get_detectron2_dataset()
        unique_image_statistics = self.get_unique_count_of_categories()
        print(unique_image_statistics)
        merged_categories = self._coco_files[0].merge_categories()
        print(merged_categories)
        amount_per_category = dict((category_id, 0) for category_id in merged_categories.values())
        for data_category_label_count in dataset_dicts:
            for annotation in data_category_label_count["annotations"]:
                current_category = annotation["category_id"]
                count = amount_per_category[current_category]
                amount_per_category[current_category] = count + 1

        amount_per_category = sorted(amount_per_category.items(), key=itemgetter(1), reverse=True)
        category_sort_order = [i for i, _ in amount_per_category]
        print("merged dataset: ", amount_per_category)
        sum_all_labels = sum(list(zip(*amount_per_category))[1])
        print("raw categories count all", sum_all_labels)

        print("Count Images: ", len(dataset_dicts))

        labels = [next((category_name for category_name, id in merged_categories.items() if id == category_id), None)
                  for category_id, count in amount_per_category]
        data_category_label_count = [count for category, count in amount_per_category]

        fig_label_count, ax_label_count = plt.subplots(figsize=(15, 5))

        rects_label_count = ax_label_count.bar(labels, data_category_label_count, width=0.5, color='gray',
                                               label="number of segmentations")

        plt.xticks(rotation=60)
        ax_label_count.set_title(
            f"Segmentations for {self.data_type} (Total sum: {sum_all_labels},Sum of Images: {len(dataset_dicts)})",
            fontsize=12)

        # ax_label_count.set_title(
        #    f"Segmentations: {self.data_type}",
        #    fontsize=14)
        ax_label_count.tick_params(axis='both', which='major', labelsize=14)
        ax_label_count.legend(fontsize=14)

        unique_image_statistics = sorted(unique_image_statistics.items(), key=itemgetter(1), reverse=True)

        sum_all_tracks = sum(list(zip(*unique_image_statistics))[1])

        amount_per_category_unique = [(category_id, count) for category_id, count in unique_image_statistics]

        # amount_per_category_unique = [(category_unique[0], category_unique[1]) for cat_id in category_sort_order for
        #                              category_unique in amount_per_category_unique if cat_id == category_unique[0]]

        self.autolabel(rects_label_count, ax_label_count)

        labels_unique = [
            next((category_name for category_name, id in merged_categories.items() if id == category_id), None)
            for category_id, count in amount_per_category_unique]

        data_category_unique_label_count = [count for category, count in unique_image_statistics]

        fig_track_count, ax_track_count = plt.subplots(figsize=(15, 5))

        rects_track_count = ax_track_count.bar(labels_unique, data_category_unique_label_count, width=0.5,
                                               color='darkblue', label="number of tracks")

        ax_track_count.set_title(
            f"Unique Tracks for {self.data_type} (Total sum: {sum_all_tracks}, Sum of Images: {len(dataset_dicts)})",
            fontsize=12)
        # ax_track_count.set_title(
        #    f"Tracks: {self.data_type}",
        #    fontsize=14)
        ax_track_count.tick_params(axis='both', which='major', labelsize=14)
        # ax_track_count.set_yticks(np.arange(0,max(data_category_unique_label_count),20))
        ax_track_count.legend(fontsize=14)

        self.autolabel(rects_track_count, ax_track_count)

        fig_label_count.tight_layout()
        fig_track_count.tight_layout()

        plt.xticks(rotation=60)
        plt.tight_layout()
        plt.show()
        # self._coco_files[0].get_category_dict_by_id(annotation["category_id"])

    def plot_statistic_with_raw_categories(self):
        print(self._coco_files[0].categories)
        amount_per_category = dict((category["name"], 0) for category in self._coco_files[0].categories)

        for coco_file in self._coco_files:
            for category in coco_file.categories:
                result_element = [(x, y) for x, y in coco_file.result_list
                                  if x["id"] == category["id"]]
                if len(result_element) > 0:
                    #  labels.append(result_element[0][0]["name"])
                    current_value = amount_per_category[
                        result_element[0][0]["name"]]
                    amount_per_category[result_element[0][0]["name"]] = result_element[0][1] + current_value

        #  fig, ax = plt.subplots(2, figsize=(12, 12), subplot_kw=dict(aspect="equal"))

        plt.figure(figsize=(15, 12))
        amount_per_category = sorted(amount_per_category.items(), key=itemgetter(1),
                                     reverse=True)

        print("raw categories without any filtering", amount_per_category)
        sum_all_labels = sum(list(zip(*amount_per_category))[1])
        print("raw categories count all", sum_all_labels)

        plt.bar([category for category, count in amount_per_category],
                [count for category, count in amount_per_category])

        plt.show()


class COCOFile:
    """
    Main Input Logic for the labeled dataset
    with mapping and cleanup functionality
    """

    #  central mapping: raw labeled categories -> final training categories
    category_mapping = {
        "car": "car",
        "cyclist": "bicycle",
        "car trailer": "car trailer",
        "truck": "truck",
        "truck trailer": "truck trailer",
        "car-transporter": "truck",
        "motorcycle": "motorcycle",
        "bus": "bus",
        "police car": "transporter",
        "firefighter truck": "truck",
        "ambulance": "ambulance",
        "pedestrian": "person",
        "pedestrian with stroller": "pedestrian with stroller",
        "pedestrian in wheelchair": "pedestrian in wheelchair",
        "scooter": "scooter",
        "transporter": "transporter"
    }

    # These categories will be ignored in the training and not served for detectron2
    ignore_categories = ['ambulance', 'bus', 'car trailer', 'truck trailer',
                         'motorcycle', 'pedestrian with stroller', 'pedestrian in wheelchair']

    def __init__(self, dataset_path, instance_file_name="instances_default"):
        annotation_file = '{}/annotations/{}.json'.format(
            dataset_path, instance_file_name)
        self.dataset_path = dataset_path
        self.coco = COCO(annotation_file)
        self.categories = self.get_categories()

    def get_unique_count_of_categories(self):
        image_statistic = {}
        processed_track_id = {-1}
        for existing_label in self.coco.anns:
            annotation = self.coco.loadAnns(existing_label)
            track_id = annotation[0]["attributes"]["track_id"]
            if track_id not in processed_track_id:
                new_category, _ = self.get_mapped_category_by_old_id(annotation[0]["category_id"])
                current_count = image_statistic[new_category] if new_category in image_statistic else 0
                image_statistic[new_category] = current_count + 1
                processed_track_id.add(track_id)
        return image_statistic

    def merge_categories(self):
        """
        use the attribute category_mapping for returning a translation map from old to new category ids
        """
        new_category_names = sorted(list(set(self.category_mapping.values())))
        new_category_names = [category_name for category_name in new_category_names if
                              category_name not in self.ignore_categories]
        id_map = {v: i for i, v in enumerate(new_category_names)}
        return id_map

    def get_detectron2_metadata(self):
        """
        provide metadata for Metadata Catalog in detectron2
        """
        id_map = self.merge_categories()
        return sorted(id_map.keys())  # things_classes for detectron2

    def get_category_dict_by_id(self, category_id):
        return next(
            (category for category in self.categories
             if category["id"] == category_id), None)

    def get_category_name_by_new_id(self, category_id):
        id_map = self.merge_categories()
        return list(id_map.keys())[list(id_map.values()).index(category_id)]
        # new_category_name = self.category_mapping

    def get_mapped_category_by_old_id(self, category_id):
        id_map = self.merge_categories()
        old_category = self.get_category_dict_by_id(category_id)
        new_category_name = self.category_mapping.get(old_category["name"])
        return new_category_name, id_map.get(new_category_name)

    def get_old_category_ids_by_new_category(self, new_category_names):
        old_category_names = [key for key, value in self.category_mapping.items() if value in new_category_names]
        return [category["id"] for category in self.categories if category["name"] in old_category_names]

    def get_new_category_id_by_coco_origin_cat(self, coco_id):
        id_map = self.merge_categories()
        old_category = self.get_category_dict_by_id(coco_id)
        return id_map.get(old_category["name"])

    def get_relevant_images(self):
        """
        return only relevant image ids for optimize trainings
        """
        relevant_category_ids = self.get_old_category_ids_by_new_category(
            ["cyclist", "person", "scooter", 'truck', 'bus', 'transporter'])
        filtered_images_ids = set()
        for category_id in relevant_category_ids:
            relevant_images = self.get_category_count_per_image(category_id)
            relevant_images_id = [image_id for image_id, count in
                                  relevant_images[:int(len(relevant_images) * 1.0)]]
            filtered_images_ids.update(relevant_images_id)
        return filtered_images_ids

    def get_detectron2_dataset_original_coco(self):
        """
        use this Method if you want to read in the original coco 2017 (instances_train2017) dataset
        """
        dataset_dicts = []
        cat_ids = self.coco.getCatIds(catNms=["person"])
        img_ids = sorted(self.coco.imgs.keys())
        imgs = self.coco.loadImgs(img_ids)
        anns = [self.coco.imgToAnns[img_id] for img_id in img_ids]
        imgs_anns = list(zip(imgs, anns))

        ann_keys = ["bbox", "category_id"]
        num_instances_without_valid_segmentation = 0
        for img_dict, anno_dict in imgs_anns:
            if len(set([ann["category_id"] for ann in anno_dict]).intersection(cat_ids)) > 0:
                record = {}
                record["file_name"] = os.path.join(self.dataset_path, "images",
                                                   img_dict["file_name"])
                record["height"] = img_dict["height"]
                record["width"] = img_dict["width"]
                image_id = img_dict["id"]
                record["image_id"] = image_id

                objs = []

                for anno in anno_dict:
                    assert anno["image_id"] == image_id
                    assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

                    obj = {key: anno[key] for key in ann_keys if key in anno}
                    category_id = self.get_new_category_id_by_coco_origin_cat(obj["category_id"])

                    if category_id is not None:
                        segm = anno.get("segmentation", None)
                        if segm:  # either list[list[float]] or dict(RLE)
                            if isinstance(segm, dict):
                                if isinstance(segm["counts"], list):
                                    # convert to compressed RLE
                                    segm = mask_util.frPyObjects(segm, *segm["size"])
                                    bitmask = rletools.decode(segm)
                                    polygons = Mask(bitmask).polygons()
                                    polygon_points = polygons.points[0].reshape((-1))
                                    if len(polygon_points) >= 6 and len(polygon_points) % 2 == 0:
                                        segm = [polygon_points]
                                    else:
                                        continue

                            else:
                                # filter out invalid polygons (< 3 points)
                                segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                                if len(segm) == 0:
                                    num_instances_without_valid_segmentation += 1
                                    continue  # ignore this instance
                            obj["segmentation"] = segm
                            obj["bbox_mode"] = BoxMode.XYWH_ABS
                            obj["category_id"] = category_id
                            objs.append(obj)
                record["annotations"] = objs
                dataset_dicts.append(record)

        return dataset_dicts[0:2000]

    def get_detectron2_dataset(self, all_images=True, coco_category_ids=False):
        """
        returns the own labeled dataset filtered by ignored cats and with category mapping in
        the detectron2 format
        """
        dataset_dicts = []
        relevant_image_ids = self.get_relevant_images()
        metadata_coco_2017 = MetadataCatalog.get('coco_2017_train')
        for image_id in self.coco.imgs:
            if image_id in relevant_image_ids or all_images:
                image = self.coco.loadImgs(image_id)
                record = {}
                file_path = os.path.join(self.dataset_path, "images",
                                         image[0]["file_name"])
                record["file_name"] = file_path
                record["image_id"] = image_id  # Take care if this number is unique!!!
                record["height"] = image[0]["height"]
                record["width"] = image[0]["width"]
                annotations = self.coco.imgToAnns[image_id]

                objs = []
                for ann in annotations:
                    obj = {}
                    assert ann["image_id"] == image_id
                    segmentation = ann.get("segmentation", None)
                    obj["segmentation"] = segmentation
                    obj["bbox"] = ann["bbox"]
                    obj["bbox_mode"] = BoxMode.XYWH_ABS
                    new_category_name, new_category_id = self.get_mapped_category_by_old_id(ann["category_id"])
                    if coco_category_ids:
                        try:
                            new_category_id = metadata_coco_2017.thing_classes.index(new_category_name)
                        except ValueError:
                            break
                    obj["category_id"] = new_category_id
                    obj["track_id"] = ann.get("attributes", None).get("track_id")

                    if new_category_name not in self.ignore_categories:
                        objs.append(obj)

                record["annotations"] = objs
                dataset_dicts.append(record)
        return dataset_dicts

    def get_category_count_per_image(self, category_id_to_filter):
        image_statistic = {}
        for annotation_id in self.coco.anns:

            annotation = self.coco.loadAnns(annotation_id)
            if len(annotation) > 0:
                category_id = annotation[0]["category_id"]
                if category_id == category_id_to_filter:
                    image_id = annotation[0]["image_id"]
                    current_value = image_statistic[
                        image_id] if image_id in image_statistic else 0
                    image_statistic[image_id] = current_value + 1
            else:
                print("This should not happen here.")
        image_statistic = sorted(image_statistic.items(),
                                 key=itemgetter(1),
                                 reverse=True)
        return image_statistic

    def get_categories(self):
        cats = self.coco.loadCats(self.coco.getCatIds())
        nms = [cat['name'] for cat in cats]
        return cats

    def get_count_of_categories(self):
        all_label_count = 0
        result_list = []
        for existing_label in self.coco.catToImgs.keys():
            labels_in_category = len(self.coco.catToImgs[existing_label])
            all_label_count = all_label_count + labels_in_category
            result_list.append(
                (self.categories[existing_label - 1], labels_in_category))

        return result_list

    def plot_statistic(self):
        labels = []
        amount_per_category = []
        for category in self.categories:
            result_element = [(x, y) for x, y in self.result_list
                              if x["id"] == category["id"]]
            if len(result_element) > 0:
                labels.append(result_element[0][0]["name"])
                amount_per_category.append(result_element[0][1])

        fig, ax = plt.subplots(2, figsize=(6, 6), subplot_kw=dict(aspect="equal"))

        def func(pct, allvals):
            absolute = int(pct / 100. * np.sum(allvals))
            return "{:.1f}%\n{:d}".format(pct, absolute)

        ax[0].set_title("All labels including duplications")

        plt.show()


if __name__ == '__main__':
    #  kitti_file = KITTIFile()
    #  print(kitti_file.get_detectron2_dataset())
    #  dataset_path = os.path.join(os.path.dirname(__file__), '../..', 'data/train2017/')

    #  cocoFile = COCOFile(dataset_path, "instances_train2017")
    #  cocoFile.get_detectron2_dataset_original_coco()
    cocoFile = ReadCOCODatasets('training', vis=True)
    cocoFile = ReadCOCODatasets('validation', vis=True)
    cocoFile = ReadCOCODatasets('test', vis=True)
