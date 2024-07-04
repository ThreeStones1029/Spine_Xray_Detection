'''
Description: this file will be used to conver drr instance categories conver to semantic categories.
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-13 11:18:33
LastEditors: ShuaiLei
LastEditTime: 2024-07-01 07:59:14
'''
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "rtdetr_paddle"))
from vis_utils.vis_coco import VisCoCo
from pycocotools.coco import COCO
from datetime import datetime


def instance_categories_conver_to_semantic_categories(instance_json_path, semantic_json_path):
    """
    The function will be used to conver drr instance categories to semantic categories.
    param: instance_json_path: The instance json file.
    param: semantic_json_path: The convered semantic json file.
    """
    if not os.path.exists(instance_json_path):
        raise ValueError("json path `{}` does not exists. ".format(instance_json_path))
    with open(instance_json_path, "r") as f:
        dataset = json.load(f)
    cat_id2cat_name = {cat["id"] :cat["name"] for cat in dataset["categories"]}
    print(cat_id2cat_name)
    for ann in dataset["annotations"]:
        cat_name_initial = cat_id2cat_name[ann["category_id"]].upper()[0]
        if cat_name_initial == "L" or cat_name_initial == "C" or cat_name_initial == "T":
            ann["category_id"] = 0
            ann["category_name"] = "vertebrae"
    dataset["categories"] = [{
                                "id": 0,
                                "name": "vertebrae"
                             }]
    with open(semantic_json_path, "w", encoding='utf-8') as w:
        json.dump(dataset, w)
    print("转换成功")


def fracture_categories_conver_to_semantic_categories(fracture_json_path, semantic_json_path):
    """
    The function will be used to conver drr fracture categories to semantic categories.
    param: fracture_json_path: The fracture json file.
    param: semantic_json_path: The converd semantic json file.
    """
    if not os.path.exists(fracture_json_path):
        raise ValueError("json path `{}` does not exists. ".format(fracture_json_path))
    with open(fracture_json_path, "r") as f:
        dataset = json.load(f)
    cat_id2cat_name = {cat["id"] :cat["name"] for cat in dataset["categories"]}
    print(cat_id2cat_name)
    for ann in dataset["annotations"]:
        cat_name = cat_id2cat_name[ann["category_id"]]
        if cat_name == "fracture" or cat_name == "normal":
            ann["category_id"] = 0
            ann["category_name"] = "vertebrae"
    dataset["categories"] = [{
                                "id": 0,
                                "name": "vertebrae"
                             }]
    with open(semantic_json_path, "w", encoding='utf-8') as w:
        json.dump(dataset, w)
    print("转换成功")


if __name__ == "__main__":
    # instance_categories_conver_to_semantic_categories("datasets/Fracture_dataset/annotations/gt_bbox.json",
    #                                                   "datasets/Fracture_dataset/annotations/semantic.json")
    # vis = VisCoCo(annotation_file="datasets/Fracture_dataset/annotations/semantic.json", 
    #               images_folder="datasets/Fracture_dataset/images", 
    #               bbox_vis_folder="datasets/Fracture_dataset/vis").visualize_bboxes_in_images()
    # vis = VisCoCo(annotation_file="datasets/Fracture_dataset/annotations/fracture_gt_bbox.json",
    #               images_folder="datasets/Fracture_dataset/images", 
    #               bbox_vis_folder="datasets/Fracture_dataset/vis_fracture").visualize_bboxes_in_images()
    fracture_categories_conver_to_semantic_categories("datasets/LA_xray_fracture/annotations/fracture_bbox_train.json",
                                                      "datasets/LA_xray_fracture/annotations/semantic_bbox_train.json")
    fracture_categories_conver_to_semantic_categories("datasets/LA_xray_fracture/annotations/fracture_bbox_val.json",
                                                      "datasets/LA_xray_fracture/annotations/semantic_bbox_val.json")
    # vis = VisCoCo(annotation_file="datasets/Fracture_dataset/annotations/fracture_bbox_test.json", 
    #               images_folder="datasets/Fracture_dataset/test", 
    #               bbox_vis_folder="datasets/Fracture_dataset/vis").visualize_bboxes_in_images()