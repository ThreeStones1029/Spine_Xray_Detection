'''
Descripttion: 
version: 
Author: ShuaiLei
Date: 2023-08-30 10:11:03
LastEditors: ShuaiLei
LastEditTime: 2024-04-13 11:13:32
'''
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "rtdetr_paddle"))
from vis_utils.vis_coco import VisCoCo
from pycocotools.coco import COCO
from datetime import datetime


def instance_label_to_semantic_label(instance_json_path, semantic_json_path):
    if not os.path.exists(instance_json_path):
        raise ValueError("json path `{}` does not exists. ".format(instance_json_path))
    with open(instance_json_path, "r") as f:
        dataset = json.load(f)
    dataset['info'] = {
                        "info": {
                            "description": "This dataset is labeled as visible to the human eye and labeled: vertebrae,pelvis, rib, bone_cement",
                            "contribute": "Shuai lei",
                            "version": "1.0",
                            "date": datetime.today().strftime('%Y-%m-%d')
                        }}
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
                             },
                             {
                                "id": 12,
                                "name": "pelvis"
                             },
                             {
                                "id": 25,
                                "name": "bone_cement"
                             },
                             {
                                "id": 26,
                                "name": "rib"
                             }
                            ]
    with open(semantic_json_path, "w", encoding='utf-8') as w:
        json.dump(dataset, w)
    print("转换成功")


class ModifyCatId(COCO):
    def __init__(self, annotation_file, modify_annotation_file):
        super(ModifyCatId, self).__init__(annotation_file)
        self.modify_annotation_file = modify_annotation_file
        self.modify_dataset = dict()
        self.modify_dataset_info = list()
        self.modify_dataset_categories = list()
        self.modify_dataset_images = list()
        self.modify_dataset_annotations = list()
        self.cat_name2cat_id = dict()
        self.cat_id2cat_name = dict()

    def modify_instance(self):
        self.add_info()
        self.add_images()
        self.modify_instance_categories()
        self.modify_annotations()
        self.save_json()

    def modify_semantic(self):
        self.add_info()
        self.add_images()
        self.modify_semantic_categories()
        self.modify_annotations()
        self.save_json()


    def add_info(self):
        """加入info"""
        self.modify_dataset_info = self.dataset["info"]
        self.modify_dataset["info"] = self.modify_dataset_info


    def add_images(self):
        self.modify_dataset_images = self.dataset["images"]
        self.modify_dataset["images"] = self.modify_dataset_images


    def modify_instance_categories(self):
        """加入合并后的具体类型信息"""
        self.modify_dataset_categories.append({"id": 1,
                                        "name": "Pelvis", 
                                        "supercategory": "vertebrae"})
        for i in range(2, 7):
            self.modify_dataset_categories.append({"id": i,
                                            "name": "L" + str(7-i),
                                            "supercategory": "vertebrae"}) 
        for i in range(7, 19):
            self.modify_dataset_categories.append({"id": i,
                                            "name": "T" + str(19-i),
                                            "supercategory": "vertebrae"}) 
        for i in range(19, 26):
            self.modify_dataset_categories.append({"id": i,
                                            "name": "C" + str(26-i),
                                            "supercategory": "vertebrae"})  
        self.modify_dataset['categories'] = self.modify_dataset_categories
        for cat in self.modify_dataset['categories']:
            self.cat_name2cat_id[cat["name"]] = cat["id"]
            self.cat_id2cat_name[cat["id"]] = cat["name"]


    def modify_semantic_categories(self):
        self.modify_dataset_categories = [{
                                            "id": 1,
                                            "name": "vertebrae"
                                        },
                                        {
                                            "id": 2,
                                            "name": "pelvis"
                                        },
                                        {
                                            "id": 3,
                                            "name": "bone_cement"
                                        },
                                        {
                                            "id": 4,
                                            "name": "rib"
                                        }]

        self.modify_dataset['categories'] = self.modify_dataset_categories
        for cat in self.modify_dataset['categories']:
            self.cat_name2cat_id[cat["name"]] = cat["id"]
            self.cat_id2cat_name[cat["id"]] = cat["name"]


    def modify_annotations(self):
        cat_id2cat_name = {cat["id"]: cat["name"] for cat in self.dataset["categories"]}
        print(cat_id2cat_name)
        for ann in self.dataset["annotations"]:
            category_id = self.cat_name2cat_id[cat_id2cat_name[ann["category_id"]]]
            category_name = cat_id2cat_name[ann["category_id"]]
            ann["category_id"] = category_id
            ann["category_name"] = category_name
            self.modify_dataset_annotations.append(ann)
        self.modify_dataset["annotations"] = self.modify_dataset_annotations


    def save_json(self):
        """结果保存"""
        with open(self.modify_annotation_file, "w") as f:
            json.dump(self.modify_dataset, f)



if __name__ == "__main__":
    # LY 20231105
    instance_label_to_semantic_label("datasets/LY20231105/annotations/instance_all.json",
                                       "datasets/LY20231105/annotations/semantic.json")
    # 修改id
    ModifyCatId(annotation_file="datasets/LY20231105/annotations/instance.json",
                modify_annotation_file="datasets/LY20231105/annotations/instance.json").modify_instance()
    ModifyCatId(annotation_file="datasets/LY20231105/annotations/semantic.json",
                modify_annotation_file="datasets/LY20231105/annotations/semantic.json").modify_semantic()
    # 可视化
    vis = VisCoCo(annotation_file="datasets/LY20231105/annotations/instance.json", 
                    images_folder="datasets/LY20231105/images", 
                    save_images_folder="datasets/LY20231105/vis_instance").visualize_images()
    vis = VisCoCo(annotation_file="datasets/LY20231105/annotations/semantic.json", 
                    images_folder="datasets/LY20231105/images", 
                    save_images_folder="datasets/LY20231105/vis_semantic").visualize_images()
    
    # TD20240117
    # instance_label_to_semantic_label("datasets/TD20240117/annotations/instance_all.json",
    #                                    "datasets/TD20240117/annotations/semantic.json")
    # # 修改id
    # ModifyCatId(annotation_file="datasets/TD20240117/annotations/instance.json",
    #             modify_annotation_file="datasets/TD20240117/annotations/instance.json").modify_instance()
    # ModifyCatId(annotation_file="datasets/TD20240117/annotations/semantic.json",
    #             modify_annotation_file="datasets/TD20240117/annotations/semantic.json").modify_semantic()
    # # 可视化
    # vis = VisCoCo(annotation_file="datasets/TD20240117/annotations/instance.json", 
    #                 images_folder="datasets/TD20240117/images", 
    #                 save_images_folder="datasets/TD20240117/vis_instance").visualize_images()
    # vis = VisCoCo(annotation_file="datasets/TD20240117/annotations/semantic.json", 
    #                 images_folder="datasets/TD20240117/images", 
    #                 save_images_folder="datasets/TD20240117/vis_semantic").visualize_images()