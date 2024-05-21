'''
Descripttion: this file is used to blend x_ray to generate blended dataset(术中和术前)
version: 
Author: ShuaiLei
Date: 2023-11-06 11:31:56
LastEditors: ShuaiLei
LastEditTime: 2024-03-12 10:22:59
'''
from pycocotools.coco import COCO
import json
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "rtdetr_paddle"))
from vis_utils.vis_coco import VisCoCo


class Gen_Blend_X_Ray_Dataset:
    def __init__(self, annotation_file_list, blend_annotation_file, blend_start_image_id, blend_start_ann_id):
        """
        param: self.blend_dataset,混合后的标注文件
        param: self.blend_dataset_info,混合后的info
        param: self.blend_dataset_images,混合后的image
        param: self.blend_dataset_categories,混合后的categories
        param: self.blend_dataset_annotations,混合后的annotations
        param: self.dataset1,需要混合的数据集标注1
        param: self.dataset2,需要混合的数据集标注2
        param: self.blend_annotation_file,混合后的标注
        param: self.cat_name2cat_id, 类别名到类别id映射
        param: self.cat_id2cat_name, 类别id到类别名映射
        param: self.blend_image_id,图像id
        param: self.blend_ann_id,ann的id
        """
        # self.coco1 = COCO(annotation_file1)
        # self.coco2 = COCO(annotation_file2)
        self.blend_dataset = dict()
        self.blend_dataset_info = dict()
        self.blend_dataset_images = list()
        self.blend_dataset_categories = list()
        self.blend_dataset_annotations = list()
        self.annotation_file_list = annotation_file_list
        self.blend_annotation_file = blend_annotation_file
        self.cat_name2cat_id = dict()
        self.cat_id2cat_name = dict()
        self.blend_image_id = blend_start_image_id
        self.blend_ann_id = blend_start_ann_id


    def blend_two_instance_dataset(self):
        """混合具体标签标注"""
        self.add_info()
        self.add_instance_categories()
        self.blend_images_and_annotations()
        self.save_json()


    def blend_two_semantic_dataset(self):
        """混合不具体标签标注"""
        self.add_info()
        self.add_semantic_categories()
        self.blend_images_and_annotations()
        self.save_json()


    def add_info(self):
        """加入info"""
        self.blend_dataset_info = {"description": "This dataset is labeled as visible to the human eye and labeled: C1-L5",
                                    "contribute": "Shuai lei",
                                    "version": "1.0",
                                    "date": datetime.today().strftime('%Y-%m-%d')}
        self.blend_dataset['info'] = self.blend_dataset_info


    def add_instance_categories(self):
        """加入合并后的具体类型信息"""
        for i in range(1, 6):
            self.blend_dataset_categories.append({"id": i,
                                                  "name": "L" + str(6-i),
                                                  "supercategory": "vertebrae"}) 
        for i in range(6, 18):
            self.blend_dataset_categories.append({"id": i,
                                                    "name": "T" + str(18-i),
                                                    "supercategory": "vertebrae"}) 
        for i in range(18, 25):
            self.blend_dataset_categories.append({"id": i,
                                                    "name": "C" + str(25-i),
                                                    "supercategory": "vertebrae"})  
        self.blend_dataset['categories'] = self.blend_dataset_categories
        for cat in self.blend_dataset['categories']:
            self.cat_name2cat_id[cat["name"]] = cat["id"]
            self.cat_id2cat_name[cat["id"]] = cat["name"]


    def add_semantic_categories(self):
        """加入不具体的类别信息"""
        self.blend_dataset_categories = [{
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
        self.blend_dataset['categories'] = self.blend_dataset_categories
        for cat in self.blend_dataset['categories']:
            self.cat_name2cat_id[cat["name"]] = cat["id"]
            self.cat_id2cat_name[cat["id"]] = cat["name"]

    
    def blend_images_and_annotations(self):
        """混合所有图像与ann"""
        for annotation_file in self.annotation_file_list:
            coco = COCO(annotation_file)
            dataset_name = annotation_file.split("/")[1]
            cat_id2cat_name = {cat["id"]: cat["name"] for cat in coco.dataset["categories"]}
            for image_info in coco.dataset["images"]:
                self.add_image_and_anns(dataset_name, coco, image_info, cat_id2cat_name)

        self.blend_dataset["images"] = self.blend_dataset_images
        self.blend_dataset["annotations"] = self.blend_dataset_annotations


    def add_image_and_anns(self, dataset_name, dataset_coco, image_info, cat_id2cat_name):
        """加入单张图片信息与anns"""
        self.blend_image_id += 1
        blend_image_info = {"id": self.blend_image_id,
                            "file_name": image_info["file_name"],
                            "width": image_info["width"],
                            "height": image_info["height"],
                            "dataset": dataset_name}
        self.blend_dataset_images.append(blend_image_info)

        for ann in dataset_coco.imgToAnns[image_info["id"]]:
            self.blend_ann_id += 1
            blend_ann = {"id": self.blend_ann_id,
                        "image_id": self.blend_image_id,
                        "iscrowd": 0,
                        "category_id": self.cat_name2cat_id[cat_id2cat_name[ann["category_id"]]],
                        "category_name": cat_id2cat_name[ann["category_id"]],
                        "bbox": ann["bbox"],
                        "area": ann["area"]}
            self.blend_dataset_annotations.append(blend_ann)
        

    def save_json(self):
        """结果保存"""
        with open(self.blend_annotation_file, "w") as f:
            json.dump(self.blend_dataset, f)



def gen_instance_intraoperative():
    # 混合术中数据集
    train = Gen_Blend_X_Ray_Dataset(annotation_file_list=["datasets/Final_instance_Label_Modify/annotations/bbox_train.json",
                                                          "datasets/Final_instance_Label_Modify/annotations/bbox_val.json",
                                                          "datasets/Final_instance_Label_Modify/annotations/bbox_test.json",
                                                          "datasets/LY20231105/annotations/instance.json",
                                                          "datasets/XJT20231101/annotations/instance.json"], 
                                    blend_annotation_file="datasets/xray20240119/annotations/train_instance.json", 
                                    blend_start_image_id=0, 
                                    blend_start_ann_id=0)
    train.blend_two_instance_dataset()
    train_images_id = len(train.blend_dataset["images"])
    train_anns_id = len(train.blend_dataset["annotations"])
    # 混合术中X线片测试验证集
    val = Gen_Blend_X_Ray_Dataset(annotation_file_list=["datasets/TD20240117/annotations/instance.json"],
                                  blend_annotation_file="datasets/xray20240119/annotations/val_instance.json", 
                                  blend_start_image_id=train_images_id, 
                                  blend_start_ann_id=train_anns_id)
    val.blend_two_instance_dataset()
    
    # 可视化
    vis_train = VisCoCo(annotation_file="datasets/xray20240119/annotations/train_instance.json", 
                        images_folder="datasets/xray20240119/train", 
                        save_images_folder="datasets/xray20240119/vis_train_instance")
    vis_train.visualize_images()
    # 可视化
    vis_val = VisCoCo(annotation_file="datasets/xray20240119/annotations/val_instance.json", 
                            images_folder="datasets/xray20240119/val", 
                            save_images_folder="datasets/xray20240119/vis_val_instance")
    vis_val.visualize_images()


def gen_semantic_intraopenrative():
    train = Gen_Blend_X_Ray_Dataset(annotation_file_list=["datasets/Final_semantic_Label_Modify/annotations/bbox_train.json",
                                                          "datasets/Final_semantic_Label_Modify/annotations/bbox_val.json",
                                                          "datasets/Final_semantic_Label_Modify/annotations/bbox_test.json",
                                                          "datasets/LY20231105/annotations/semantic.json",
                                                          "datasets/XJT20231101/annotations/semantic.json"], 
                                    blend_annotation_file="datasets/xray20240119/annotations/train_semantic.json", 
                                    blend_start_image_id=0, 
                                    blend_start_ann_id=0)
    train.blend_two_semantic_dataset()
    train_images_id = len(train.blend_dataset["images"])
    train_anns_id = len(train.blend_dataset["annotations"])
    val = Gen_Blend_X_Ray_Dataset(annotation_file_list=["datasets/TD20240117/annotations/semantic.json"], 
                                       blend_annotation_file="datasets/xray20240119/annotations/val_semantic.json", 
                                       blend_start_image_id=train_images_id, 
                                       blend_start_ann_id=train_anns_id)
    val.blend_two_semantic_dataset()
    
    vis_train = VisCoCo(annotation_file="datasets/xray20240119/annotations/train_semantic.json", 
                        images_folder="datasets/xray20240119/train", 
                        save_images_folder="datasets/xray20240119/vis_train_semantic")
    vis_train.visualize_images()
    vis_val= VisCoCo(annotation_file="datasets/xray20240119/annotations/val_semantic.json", 
                            images_folder="datasets/xray20240119/val", 
                            save_images_folder="datasets/xray20240119/vis_val_semantic")
    vis_val.visualize_images()


if __name__ == "__main__":
    gen_instance_intraoperative()
    gen_semantic_intraopenrative()