'''
Description: 一些数据集的json处理方法
version: 1.0
Author: ShuaiLei
Date: 2023-11-10 19:14:07
LastEditors: ShuaiLei
LastEditTime: 2024-03-12 10:23:33
'''
from copy import deepcopy
import os
import shutil
import json
from pycocotools.coco import COCO
import cv2


def load_json_file(json_path):
    with open(json_path, "r") as f:
        dataset = json.load(f)
    return dataset


def save_json_file(data, json_path):
    dirname = os.path.dirname(json_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(json_path, 'w') as f:
        json.dump(data, f)
    print(json_path, "save successfully")


def choose_images_from_coco_json(json_path, images_folder, choosed_images_folder):
    """
    param: coco_json_file_path,the choosed images coco json
    param: BUU_images_folder, the public dataset folder path
    param: choosed_BUU_images_folder, the choosed BUU images save path
    """
    dataset = load_json_file(json_path)
    for image in dataset["images"]:
        if os.path.isfile(os.path.join(images_folder, image["file_name"])):
            print(image["file_name"])
            shutil.copy(os.path.join(images_folder, image["file_name"]), os.path.join(choosed_images_folder, image["file_name"]))
    print("copy completed")


def choose_annotation_according_images(images_folder, json_path, choosed_json_path):
    """
    根据选择的图片筛选对应的标注
    """
    files = os.listdir(images_folder)
    gt = COCO(json_path)

    choosed_dataset = {"info": gt.dataset["info"], "categories": gt.dataset["categories"], "images": [], "annotations": []}
    for image in gt.dataset["images"]:
        if image["file_name"] in files:
            choosed_dataset["images"].append(image)
            for ann in gt.imgToAnns[image["id"]]:
                choosed_dataset["annotations"].append(ann)
    save_json_file(choosed_dataset, choosed_json_path)


def get_exist_categories(json_path):
    """
    获取存在的类别
    """
    dataset = load_json_file(json_path)
    catid2catname = {}
    for category in dataset["categories"]:
        catid2catname[category["id"]] = category["name"]
    exist_categories = []
    for ann in dataset["annotations"]:
        catid = ann["category_id"]
        catname = ann["category_name"] if "category_name" in ann.keys() else catid2catname[catid]
        if catname not in exist_categories:
            exist_categories.append(catname)
    return exist_categories


def find_choosed_categories_images(choosed_category_names, json_path):
    """
    找到数据集中指定类别的图片
    """
    gt = COCO(json_path)
    catid2catname = {}
    catname2catid = {}
    for category in gt.dataset["categories"]:
        catid2catname[category["id"]] = category["name"]
        catname2catid[category["name"]] = category["id"]
    for catname in choosed_category_names:
        catid = catname2catid[catname]
        image_ids = gt.catToImgs[catid]
        for image_id in image_ids:
            image = gt.imgs[image_id]
            print(image["file_name"])
            print(image["id"])


def unify_():
    pass


def split_AP_and_LA(json_path, AP_json_path, LA_json_path):
    """
    拆分AP和LA为单独的json
    """
    gt = COCO(json_path)
    AP_dataset = {"info": gt.dataset["info"],
                  "categories": gt.dataset["categories"],
                  "images": [],
                  "annotations": []}
    LA_dataset = {"info": gt.dataset["info"],
                  "categories": gt.dataset["categories"],
                  "images": [],
                  "annotations": []}
    
    for image in gt.dataset["images"]:
        imgToAnns = gt.imgToAnns[image["id"]]
        if image["type"] == "AP":
            AP_dataset["images"].append(image)
            for ann in imgToAnns:
                AP_dataset["annotations"].append(ann)
        if image["type"] == "LA":
            LA_dataset["images"].append(image)
            for ann in imgToAnns:
                LA_dataset["annotations"].append(ann)

    save_json_file(AP_dataset, AP_json_path)
    save_json_file(LA_dataset, LA_json_path)


def check_image_id_and_file_name(json_path):
    filenames = []
    dataset = load_json_file(json_path)
    for image in dataset["images"]:
        filenames.append(image["file_name"])
    return filenames


def predict_format2coco_format(predict_folder_path, predict_json_path, save_coco_json_path, template_coco_json_path):
    """
    用于转换预测标注到训练格式的标注
    """
    predict_anns = load_json_file(predict_json_path)
    template_coco_gt_dataset = load_json_file(template_coco_json_path)
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
    save_json_file(coco_format_dataset, save_coco_json_path)
 

def merge_coco_format_annotation(annotation_file_list, merge_annotation_file):
    """
    用于混合coco格式标注
    """
    dataset = load_json_file(annotation_file_list[0])
    merge_dataset = {"info": dataset["info"],
                     "categories": dataset["categories"],
                     "images": [],
                     "annotations": []}
    for annotation_file in annotation_file_list:
        dataset = load_json_file(annotation_file)
        for image in dataset["images"]:
            merge_dataset["images"].append(image)
        for ann in dataset["annotations"]:
            merge_dataset["annotations"].append(ann)
    save_json_file(merge_dataset, merge_annotation_file)


def gen_retrain_semantic_annotation_file(predict_folder_path, instance_json_path, semantic_predict_json_path, template_coco_json_path, retrain_semantic_json_path):
    """
    用于将logic推理后train_60的椎体标签,加上预测后的大类以及骨盆、骨水泥标注作为最终train_60的推理大类标注
    param: predict_folder_path:图片根目录
    param: instance_json_path:用于重新train的具体标注
    param: semantic_predict_json_path:大类的预测文件
    param: template_coco_json_path:大类模板标注
    """
    instance_dataset = load_json_file(instance_json_path)
    template_coco_gt_dataset = load_json_file(template_coco_json_path)
    semantic_predict_anns = load_json_file(semantic_predict_json_path)
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

    # j加入大类的骨盆以及肋骨预测框
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

    save_json_file(retrain_dataset, retrain_semantic_json_path)


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
    save_json_file(filter_dataset, annotation_file)


def add_miss_anns_in_semantic_predict(annotation_file, semantic_predict_json_path):
    retrain = COCO(annotation_file)
    with open(semantic_predict_json_path, "r") as f:
        semantic_predict = json.load(f)
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
    with open(annotation_file, "w") as f:
        json.dump(retrain.dataset, f)
            
       
if __name__ == "__main__":
    # exist_categories = get_exist_categories("datasets/miccai/xray/annotations/train_instance.json")
    # print(exist_categories)
    # find_choosed_categories_images(["T8"], "datasets/miccai/xray/annotations/train_instance.json")
    # choose_annotation_according_images("datasets/miccai/xray/val", 
    #                                    "datasets/miccai/xray/annotations/xray_semantic.json",
    #                                    "datasets/miccai/xray/annotations/val_semantic.json")

    # split_AP_and_LA("datasets/miccai/BUU/annotations/buu_5800.json",
    #                 "datasets/miccai/BUU/annotations/buu_5800_AP.json",
    #                 "datasets/miccai/BUU/annotations/buu_5800_LA.json")
    # find_choosed_categories_images(["T1", "T2", "T3", "T4", "T5"], "datasets/miccai/BUU/annotations/buu_5800.json")
    # choose_images_from_coco_json("datasets/miccai/buu/annotations/buu_5800.json",
    #                              "/home/jjf/paddle/PaddleDetection/dataset/buu/images",
    #                              "datasets/miccai/buu/images")   
    # semantic_filenames = check_image_id_and_file_name("datasets/miccai/xray/annotations/val_semantic.json")
    # instance_filenames = check_image_id_and_file_name("datasets/miccai/xray/annotations/val_instance.json")
    # i = 0
    # for filename in semantic_filenames:
    #     if filename in instance_filenames:
    #         i += 1
    #     else:
    #         print(filename)
    # print(i)
    choose_annotation_according_images("datasets/miccai/buu/images", 
                                       "datasets/miccai/buu/annotations/buu_5800.json",
                                       "datasets/miccai/buu/annotations/0146-M-057Y1.json")

    # 用于生成retrain instance 标注
    # predict_format2coco_format("datasets/miccai/xray/images/train_instance_60",
    #                            "datasets/miccai/xray/rtdetr101_annotations_retrain2/train_60_logic.json",
    #                            "datasets/miccai/xray/rtdetr101_annotations_retrain2/retrain_instance_60.json",
    #                            "datasets/miccai/xray/rtdetr101_annotations_retrain2/train_instance.json")
    # merge_coco_format_annotation(["datasets/miccai/xray/rtdetr101_annotations_retrain2/retrain_instance_60.json",
    #                               "datasets/miccai/xray/rtdetr101_annotations_retrain2/train_instance_20.json"],
    #                               "datasets/miccai/xray/rtdetr101_annotations_retrain2/retrain_instance.json")

    # 用于生成retrain semantic 标注
    # gen_retrain_semantic_annotation_file("datasets/miccai/xray/images/train_instance_60",
    #                                         "datasets/miccai/xray/rtdetr101_annotations_retrain2/retrain_instance_60.json",
    #                                         "datasets/miccai/xray/rtdetr101_annotations_retrain2/train_semantic_60_predict_postprocess.json",
    #                                         "datasets/miccai/xray/rtdetr101_annotations_retrain2/val_semantic.json",
    #                                         "datasets/miccai/xray/rtdetr101_annotations_retrain2/retrain_semantic_60.json")
    # filter_repet_anns("datasets/miccai/xray/rtdetr101_annotations_retrain2/retrain_semantic_60.json")
    # add_miss_anns_in_semantic_predict("datasets/miccai/xray/rtdetr101_annotations_retrain2/retrain_semantic_60.json",
    #                                      "datasets/miccai/xray/rtdetr101_annotations_retrain2/train_semantic_60_predict_postprocess.json")
    # merge_coco_format_annotation(["datasets/miccai/xray/rtdetr101_annotations_retrain2/retrain_semantic_60.json",
    #                               "datasets/miccai/xray/rtdetr101_annotations_retrain2/train_semantic_20.json"],
    #                               "datasets/miccai/xray/rtdetr101_annotations_retrain2/retrain_semantic.json")
   



