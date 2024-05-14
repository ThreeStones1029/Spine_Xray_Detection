'''
Description: 
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-05-14 05:16:28
LastEditors: ShuaiLei
LastEditTime: 2024-05-14 14:06:54
'''
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "rtdetr_paddle"))
import shutil
import numpy as np
from json_process import load_json_file, save_json_file
from vis_utils.vis_coco import VisCoCo


def split_train_dataset_to_4_folds(images_folder_path, 
                                    instance_annotation_file, 
                                    semantic_annotation_file,
                                    output_folder_path, 
                                    split_info_dict = {"fold1": 28, "fold2": 29, "fold3": 28, "fold4": 29}):
    """
    This function will be used split train dataset to 4 folds for cross valid training.
    param: train_images_folder: the train images folder.
    param: instance_annotation_file: the instance annotation json file.
    param: semantic_annotation_file: the semantic annotation json file. 
    param: output_folder_path: the output folder path.
    param: split_info_dict: the split parts number.
    """
    os.makedirs(output_folder_path, exist_ok=True)
    # 读取annotations.json文件
    instance_dataset = load_json_file(instance_annotation_file)
    semantic_dataset = load_json_file(semantic_annotation_file)
    # 提取images, annotations, categories
    # 随机打乱数据 注意需要打乱需要保持一样的打乱,因为同时提取两种标注
    instance_dataset["images"] = sorted(instance_dataset["images"], key=lambda x: x['id'])
    semantic_dataset["images"] = sorted(semantic_dataset["images"], key=lambda x: x['id'])
    np.random.seed(42)
    np.random.shuffle(instance_dataset["images"])
    np.random.seed(42)
    np.random.shuffle(semantic_dataset["images"])
    print([image["id"] for image in instance_dataset["images"]])
    print([image["id"] for image in semantic_dataset["images"]])
    start_index = 0
    end_index = 0
    def filter_annotations(annotations, image_ids):
        return [ann for ann in annotations if ann["image_id"] in image_ids]
    for split_part_name, part_number in split_info_dict.items():
        end_index += part_number
        instance_split_part_images = instance_dataset["images"][start_index:end_index]
        semantic_split_part_images = semantic_dataset["images"][start_index:end_index]
        print(start_index, end_index)
        start_index += part_number
        split_part_folder = os.path.join(output_folder_path, split_part_name)
        os.makedirs(split_part_folder, exist_ok=True)
        for img in instance_split_part_images:
            shutil.copy(os.path.join(images_folder_path, img["file_name"]), os.path.join(split_part_folder, img["file_name"]))
            print(os.path.join(split_part_folder, img["file_name"]))
        instance_split_part_annotations = filter_annotations(instance_dataset["annotations"], [img["id"] for img in instance_split_part_images])
        semantic_split_part_annotations = filter_annotations(semantic_dataset["annotations"], [img["id"] for img in semantic_split_part_images])
        instance_split_part_data = {"info": instance_dataset["info"], 
                                    "images": instance_split_part_images, 
                                    "annotations": instance_split_part_annotations, 
                                    "categories": instance_dataset["categories"]}
        semantic_split_part_data = {"info": semantic_dataset["info"], 
                                    "images": semantic_split_part_images, 
                                    "annotations": semantic_split_part_annotations, 
                                    "categories": semantic_dataset["categories"]}
        print([image["file_name"] for image in instance_split_part_data["images"]])
        print([image["file_name"] for image in semantic_split_part_data["images"]])
        save_json_file(instance_split_part_data, os.path.join(output_folder_path,"instance_bbox_" + split_part_name + ".json"))
        save_json_file(semantic_split_part_data, os.path.join(output_folder_path,"semantic_bbox_" + split_part_name + ".json"))
    print("数据集划分完成！")


def merge_train_dataset(dataset_root_folder, fold_list, train_images_folder, annotations_folder):
    """
    The function will be used to merge train dataset.
    param: dataset_root_folder: 
    param: fold_list: the choosed fold will be used to merge train dataset.
    param: train_images_folder: 
    param: annotations_folder: 
    """
    instance_fold0_dataset = load_json_file(os.path.join(dataset_root_folder, "instance_bbox_" + fold_list[0] + ".json"))
    semantic_fold0_dataset = load_json_file(os.path.join(dataset_root_folder, "semantic_bbox_" + fold_list[0] + ".json"))
    os.makedirs(train_images_folder, exist_ok=True)
    instance_merge_dataset = {"info": instance_fold0_dataset["info"],
                              "images": [],
                              "annotations": [],
                              "categories": instance_fold0_dataset["categories"]}
    semantic_merge_dataset = {"info": semantic_fold0_dataset["info"],
                              "images": [],
                              "annotations": [],
                              "categories": semantic_fold0_dataset["categories"]}
    
    for fold in fold_list:
        instance_dataset = load_json_file(os.path.join(dataset_root_folder, "instance_bbox_" + fold + ".json"))
        semantic_dataset = load_json_file(os.path.join(dataset_root_folder, "semantic_bbox_" + fold + ".json"))
        for image in instance_dataset["images"]:
            instance_merge_dataset["images"].append(image)
            shutil.copy(os.path.join(dataset_root_folder, fold, image["file_name"]), os.path.join(train_images_folder, image["file_name"]))
        instance_merge_dataset["images"] = sorted(instance_merge_dataset["images"], key=lambda x: x['id'])
        for ann in instance_dataset["annotations"]:
            instance_merge_dataset["annotations"].append(ann)
        instance_merge_dataset["annotations"] = sorted(instance_merge_dataset["annotations"], key=lambda x: x['id'])
        for image in semantic_dataset["images"]:
            semantic_merge_dataset["images"].append(image)
        semantic_merge_dataset["images"] = sorted(semantic_merge_dataset["images"], key=lambda x: x['id'])
        for ann in semantic_dataset["annotations"]:
            semantic_merge_dataset["annotations"].append(ann)
        semantic_merge_dataset["annotations"] = sorted(semantic_merge_dataset["annotations"], key=lambda x: x['id'])
    
    
    save_json_file(instance_merge_dataset, os.path.join(annotations_folder, "train_instance.json"))
    save_json_file(semantic_merge_dataset, os.path.join(annotations_folder, "train_semantic.json"))


def split_train_dataset_to_20_percentage_and_60_percentage(images_folder_path, 
                                                            instance_annotation_file, 
                                                            semantic_annotation_file,
                                                            output_folder_path, 
                                                            split_info_dict = {"train_20": 29, "train_60": 86}):
    """
    This function will be used split train dataset to 4 folds for cross valid training.
    param: train_images_folder: the train images folder.
    param: instance_annotation_file: the instance annotation json file.
    param: semantic_annotation_file: the semantic annotation json file. 
    param: output_folder_path: the output folder path.
    param: split_info_dict: the split parts number.
    """
    os.makedirs(output_folder_path, exist_ok=True)
    # 读取annotations.json文件
    instance_dataset = load_json_file(instance_annotation_file)
    semantic_dataset = load_json_file(semantic_annotation_file)
    # 提取images, annotations, categories
    # 随机打乱数据 注意需要打乱需要保持一样的打乱,因为同时提取两种标注
    instance_dataset["images"] = sorted(instance_dataset["images"], key=lambda x: x['id'])
    semantic_dataset["images"] = sorted(semantic_dataset["images"], key=lambda x: x['id'])
    np.random.seed(42)
    np.random.shuffle(instance_dataset["images"])
    np.random.seed(42)
    np.random.shuffle(semantic_dataset["images"])
    start_index = 0
    end_index = 0
    def filter_annotations(annotations, image_ids):
        return [ann for ann in annotations if ann["image_id"] in image_ids]
    for split_part_name, part_number in split_info_dict.items():
        end_index += part_number
        instance_split_part_images = instance_dataset["images"][start_index:end_index]
        semantic_split_part_images = semantic_dataset["images"][start_index:end_index]
        start_index += part_number
        split_part_folder = os.path.join(output_folder_path, split_part_name)
        os.makedirs(split_part_folder, exist_ok=True)
        for img in instance_split_part_images:
            shutil.copy(os.path.join(images_folder_path, img["file_name"]), os.path.join(split_part_folder, img["file_name"]))
            print(os.path.join(split_part_folder, img["file_name"]))
        instance_split_part_annotations = filter_annotations(instance_dataset["annotations"], [img["id"] for img in instance_split_part_images])
        semantic_split_part_annotations = filter_annotations(semantic_dataset["annotations"], [img["id"] for img in semantic_split_part_images])
        # 排序
        instance_split_part_images = sorted(instance_split_part_images, key=lambda x: x['id'])
        semantic_split_part_images = sorted(semantic_split_part_images, key=lambda x: x['id'])
        instance_split_part_annotations = sorted(instance_split_part_annotations, key=lambda x: x['id'])
        semantic_split_part_annotations = sorted(semantic_split_part_annotations, key=lambda x: x['id'])
        instance_split_part_data = {"info": instance_dataset["info"], 
                                    "images": instance_split_part_images, 
                                    "annotations": instance_split_part_annotations, 
                                    "categories": instance_dataset["categories"]}
        semantic_split_part_data = {"info": semantic_dataset["info"], 
                                    "images": semantic_split_part_images, 
                                    "annotations": semantic_split_part_annotations, 
                                    "categories": semantic_dataset["categories"]}
        percentage = split_part_name.split("_")[-1]
        save_json_file(instance_split_part_data, os.path.join(output_folder_path, "train_instance_" + percentage + ".json"))
        save_json_file(semantic_split_part_data, os.path.join(output_folder_path, "train_semantic_" + percentage + ".json"))
    print("数据集划分完成！")


if __name__ == "__main__":
    # split_train_dataset_to_4_folds("/home/RT-DETR/rtdetr_paddle/datasets/miccai/xray/images0/train_semantic", 
    #                                 "/home/RT-DETR/rtdetr_paddle/datasets/miccai/xray/annotations0/train_instance.json",
    #                                 "/home/RT-DETR/rtdetr_paddle/datasets/miccai/xray/annotations0/train_semantic.json",       
    #                                 "/home/RT-DETR/rtdetr_paddle/datasets/miccai/xray")

    # merge_train_dataset("/home/RT-DETR/rtdetr_paddle/datasets/miccai/xray",
    #                     ["fold0",
    #                      "fold1",
    #                      "fold2",
    #                      "fold3"],
    #                      train_images_folder="/home/RT-DETR/rtdetr_paddle/datasets/miccai/xray/images4/train",
    #                      annotations_folder="/home/RT-DETR/rtdetr_paddle/datasets/miccai/xray/annotations4")
    vis = VisCoCo(annotation_file="/home/RT-DETR/rtdetr_paddle/datasets/miccai/xray/annotations1/train_instance_20.json", 
                  images_folder="/home/RT-DETR/rtdetr_paddle/datasets/miccai/xray/images1/train_20", 
                  bbox_vis_folder="/home/RT-DETR/rtdetr_paddle/datasets/miccai/xray/images1/gt")

    vis.visualize_bboxes_in_images()

    # split_train_dataset_to_20_percentage_and_60_percentage("/home/RT-DETR/rtdetr_paddle/datasets/miccai/xray/images1/train", 
    #                                                         "/home/RT-DETR/rtdetr_paddle/datasets/miccai/xray/annotations1/train_instance.json",
    #                                                         "/home/RT-DETR/rtdetr_paddle/datasets/miccai/xray/annotations1/train_semantic.json",       
    #                                                         "/home/RT-DETR/rtdetr_paddle/datasets/miccai/xray/images1",
    #                                                         split_info_dict={"train_20": 29, "train_60":86})