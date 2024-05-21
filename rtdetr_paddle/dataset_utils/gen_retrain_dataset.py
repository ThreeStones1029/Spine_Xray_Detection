'''
Description: 
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-11 03:18:39
LastEditors: ShuaiLei
LastEditTime: 2024-05-16 12:05:46
'''
'''
Description: this file will be used to conver logical reason result to pseudo-label which will be used to retrain instance and semantic network
version: 1.0
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-03-14 16:08:58
LastEditors: ShuaiLei
LastEditTime: 2024-03-14 21:41:48
'''
import cv2
import os
import json
from pycocotools.coco import COCO


def load_json(json_path):
    with open(json_path, "r") as f:
        json_data = json.load(f)
    return json_data


def save_json(data, save_json_path):
    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
    with open(save_json_path, 'w', encoding='utf-8') as f_write:
        json.dump(data, f_write)
    print(save_json_path, "save successfully!")


def predict_format2coco_format(predict_folder_path, predict_json_path, save_coco_json_path, template_coco_json_path):
    """
    Transform predicted annotations to annotations in the training format
    param: predict_folder_path(str): the images folder used to run logical reasoning
    param: predict_json_path(str): thr logical reasoning result
    param: save_coco_jsaon_path(str): the convered json file save path
    param: template_coco_json_path(str): the gt annotations tempalte
    return: None
    """
    predict_anns = load_json(predict_json_path)
    template_coco_gt_dataset = load_json(template_coco_json_path)
    coco_format_dataset = {"info": template_coco_gt_dataset["info"],
                           "categories": template_coco_gt_dataset["categories"],
                           "images": [],
                           "annotations": []}
    ann_id = 10000  # 指定开始的id，避免与原来的id冲突
    exist_file_name_list = []
    for ann in predict_anns:
        ann_id += 1
        new_ann = {"id": ann_id,
                   "image_id": ann["image_id"],
                   "iscrowd": 0,
                   "category_id": ann["category_id"],
                   "category_name": ann["category_name"],
                   "bbox": ann["bbox"],
                   "area": ann["bbox"][2] * ann["bbox"][3]}
        coco_format_dataset["annotations"].append(new_ann)

        img = cv2.imread(os.path.join(predict_folder_path, ann["file_name"]))
        width, height, channels = img.shape
        if ann["file_name"] not in exist_file_name_list:
            new_image = {"id": ann["image_id"],
                         "file_name": ann["file_name"],
                         "width": width,
                         "height": height,
                         "dataset": "logic"}
            exist_file_name_list.append(ann["file_name"])
            coco_format_dataset["images"].append(new_image)
    save_json(coco_format_dataset, save_coco_json_path)


def merge_coco_format_annotation(annotation_file_list, merge_annotation_file):
    """
    Used for merge coco format annotation
    the logical result conver to coco format annotation, then need to merge truth label and pseudo-label
    param: annotation_file_list(str): label files need to be merged to one json file
    param: merge_annotation_file(str): the merged json file save path
    return None
    """
    dataset = load_json(annotation_file_list[0])
    merge_dataset = {"info": dataset["info"],
                     "categories": dataset["categories"],
                     "images": [],
                     "annotations": []}
    for annotation_file in annotation_file_list:
        dataset = load_json(annotation_file)
        for image in dataset["images"]:
            merge_dataset["images"].append(image)
        for ann in dataset["annotations"]:
            merge_dataset["annotations"].append(ann)
    save_json(merge_dataset, merge_annotation_file)


def gen_retrain_semantic_annotation_file(predict_folder_path, instance_json_path, semantic_predict_json_path, template_coco_json_path, retrain_semantic_json_path):
    """
    用于将logic推理后train_60的椎体标签,加上预测后的大类以及骨盆、骨水泥标注作为最终train_60的推理大类标注
    param: predict_folder_path:图片根目录
    param: instance_json_path:用于重新train的具体标注
    param: semantic_predict_json_path:大类的预测文件
    param: template_coco_json_path:大类模板标注
    """
    instance_dataset = load_json(instance_json_path)
    template_coco_gt_dataset = load_json(template_coco_json_path)
    semantic_predict_anns = load_json(semantic_predict_json_path)
    retrain_dataset = {"info": template_coco_gt_dataset["info"],
                       "categories": template_coco_gt_dataset["categories"],
                       "images": [],
                       "annotations": []}
    retrain_dataset["images"] = instance_dataset["images"]
    exist_file_name_list = []
    for image in retrain_dataset["images"]:
        exist_file_name_list.append(image["file_name"])
    catname2catid = {}
    for category in template_coco_gt_dataset["categories"]:
        catname2catid[category["name"]] = category["id"]
    # 加入具体的椎体推理框，修改具体标签为大类vertebrae
    max_ann_id = 0 # 记录最大的id，用于记录骨盆肋骨id，防止冲突
    for ann in instance_dataset["annotations"]:
        ann["category_name"] = "vertebrae"
        ann["category_id"] = catname2catid["vertebrae"]
        retrain_dataset["annotations"].append(ann)
        if ann["id"] > max_ann_id:
            max_ann_id = ann["id"]

    # 加入大类的骨盆以及肋骨预测框
    for ann in semantic_predict_anns:
        if ann["file_name"] not in exist_file_name_list:
            img = cv2.imread(os.path.join(predict_folder_path, ann["file_name"]))
            width, height, channels = img.shape
            new_image = {"id": ann["image_id"],
                         "file_name": ann["file_name"],
                         "width": width,
                         "height": height,
                         "dataset": "no instance predict"}
            exist_file_name_list.append(ann["file_name"])
            retrain_dataset["images"].append(new_image)

        new_ann = {}
        if (ann["category_name"] == "pelvis" or ann["category_name"] == "rib" or ann["category_name"] == "bone_cement"):
            max_ann_id += 1
            new_ann["id"] = max_ann_id
            new_ann["image_id"] = ann["image_id"]
            new_ann["category_name"] = ann["category_name"]
            new_ann["category_id"] = catname2catid[ann["category_name"]]
            new_ann["bbox"] = ann["bbox"]
            new_ann["area"] = ann["bbox"][2] * ann["bbox"][3]
            new_ann["iscrowd"] = 0
            retrain_dataset["annotations"].append(new_ann)
    print(len(retrain_dataset["images"]))
    save_json(retrain_dataset, retrain_semantic_json_path)


def get_bbox_center(ann):
    x = ann["bbox"][0] + ann["bbox"][2] / 2
    y = ann["bbox"][1] + ann["bbox"][3] / 2
    return x, y 


def judgment_bboxes_intersection(ann1, ann2):
    w1, h1 = ann1['bbox'][2], ann1['bbox'][3]
    w2, h2 = ann2['bbox'][2], ann2['bbox'][3]
    bbox1_center_x, bbox1_center_y = get_bbox_center(ann1)
    bbox2_center_x, bbox2_center_y = get_bbox_center(ann2)
    bbox_center_distance_x = abs(bbox1_center_x - bbox2_center_x)
    bbox_center_distance_y = abs(bbox1_center_y - bbox2_center_y)
    if bbox_center_distance_x < (w1 + w2)/2.0 and bbox_center_distance_y < (h1 + h2)/2.0:
        return True
    else:
        return False


def remove_overlap_anns(anns):
    for i, ann1 in enumerate(anns):
        for j, ann2 in enumerate(anns):
            if i != j and judgment_bboxes_intersection(ann1, ann2):
                anns.remove(ann2)
    return anns
    
            
def filter_repet_anns(annotation_file):
    gt = COCO(annotation_file)
    filter_dataset = {"info": gt.dataset["info"],
                      "categories": gt.dataset["categories"],
                      "images": gt.dataset["images"],
                      "annotations": []}
    for image in gt.dataset["images"]:
        pelvis_anns = []
        rib_anns = []
        bone_cement_anns = []
        vertebrae_anns = []
        for ann in gt.imgToAnns[image["id"]]:
            # 椎体框直接加入
            if ann["category_name"] == "vertebrae":
                vertebrae_anns.append(ann)
            if ann["category_name"] == "pelvis":
                pelvis_anns.append(ann)
            if ann["category_name"] == "rib":
                rib_anns.append(ann)
            if ann["category_name"] == "bone_cement":
                bone_cement_anns.append(ann)
        filter_pelvis_anns = remove_overlap_anns(pelvis_anns)
        filter_rib_anns = remove_overlap_anns(rib_anns)
        filter_bone_cement_anns = remove_overlap_anns(bone_cement_anns)
        for ann in filter_pelvis_anns:
            filter_dataset["annotations"].append(ann)
        for ann in filter_rib_anns:
            filter_dataset["annotations"].append(ann)
        for ann in filter_bone_cement_anns:
            filter_dataset["annotations"].append(ann)
        for ann in vertebrae_anns:
            filter_dataset["annotations"].append(ann)
    save_json(filter_dataset, annotation_file)


def add_miss_anns_in_semantic_predict(annotation_file, semantic_predict_json_path):
    retrain = COCO(annotation_file)
    semantic_predict = load_json(semantic_predict_json_path)
    max_ann_id = 0
    for ann in retrain.dataset["annotations"]:
        if ann["id"] > max_ann_id:
            max_ann_id = ann["id"]
    catname2catid = {}
    for category in retrain.dataset["categories"]:
        catname2catid[category["name"]] = category["id"]
    for image in retrain.dataset["images"]:
        if len(retrain.imgToAnns[image["id"]]) == 0:
            print(image["file_name"])
            for ann in semantic_predict:
                new_ann = {}
                if ann["image_id"] == image["id"] or ann["file_name"] == image["file_name"]:
                    max_ann_id += 1
                    new_ann["id"] = max_ann_id
                    new_ann["image_id"] = ann["image_id"]
                    new_ann["category_name"] = ann["category_name"]
                    new_ann["category_id"] = catname2catid[ann["category_name"]]
                    new_ann["bbox"] = ann["bbox"]
                    new_ann["area"] = ann["bbox"][2] * ann["bbox"][3]
                    new_ann["iscrowd"] = 0
                    retrain.dataset["annotations"].append(new_ann)
    save_json(retrain.dataset, annotation_file)


if __name__ == "__main__":
    # 用于生成retrain instance 标注
    # predict_format2coco_format(predict_folder_path="datasets/miccai/xray/images4/train_60",
    #                            predict_json_path="datasets/miccai/xray/rtdetr50_annotations_retrain4/train_60_logic.json",
    #                            save_coco_json_path="datasets/miccai/xray/rtdetr50_annotations_retrain4/retrain_instance_60.json",
    #                            template_coco_json_path="datasets/miccai/xray/rtdetr50_annotations_retrain4/train_instance.json")
    # merge_coco_format_annotation(["datasets/miccai/xray/rtdetr50_annotations_retrain4/retrain_instance_60.json",
    #                               "datasets/miccai/xray/rtdetr50_annotations_retrain4/train_instance_20.json"],
    #                               "datasets/miccai/xray/rtdetr50_annotations_retrain4/retrain_instance.json")

    # 用于生成retrain semantic 标注
    gen_retrain_semantic_annotation_file("datasets/miccai/xray/images4/train_60",
                                            "datasets/miccai/xray/rtdetr50_annotations_retrain4/retrain_instance_60.json",
                                            "datasets/miccai/xray/rtdetr50_annotations_retrain4/train_semantic_60_predict_postprocess.json",
                                            "datasets/miccai/xray/rtdetr50_annotations_retrain4/val_semantic.json",
                                            "datasets/miccai/xray/rtdetr50_annotations_retrain4/retrain_semantic_60.json")
    filter_repet_anns("datasets/miccai/xray/rtdetr50_annotations_retrain4/retrain_semantic_60.json")
    add_miss_anns_in_semantic_predict("datasets/miccai/xray/rtdetr50_annotations_retrain4/retrain_semantic_60.json",
                                         "datasets/miccai/xray/rtdetr50_annotations_retrain4/train_semantic_60_predict_postprocess.json")
    merge_coco_format_annotation(["datasets/miccai/xray/rtdetr50_annotations_retrain4/retrain_semantic_60.json",
                                  "datasets/miccai/xray/rtdetr50_annotations_retrain4/train_semantic_20.json"],
                                  "datasets/miccai/xray/rtdetr50_annotations_retrain4/retrain_semantic.json")