'''
Descripttion: this file is used to cut BUU dataset's images to images of the intraoperative X-ray field size.
version: 1.0
Author: ShuaiLei
Date: 2023-11-04 17:43:17
LastEditors: ShuaiLei
LastEditTime: 2024-02-26 15:37:24
'''
import json
from pycocotools.coco import COCO
from datetime import datetime
from PIL import Image
import os
import sys
import numpy as np
import random
import multiprocessing
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "rtdetr_paddle"))


class Cut_Image(COCO):
    def __init__(self, annotation_file=None, images_folder="images", is_flip=False, save_path=None, save_json_name="cut.json"):
        super(Cut_Image, self).__init__(annotation_file)
        """
        this file is used to cut BUU dataset's images to images of the intraoperative X-ray field size.
        param self.images_folder 选择的需要裁剪的图片文件夹
        param self.save_path 裁剪后图片的保存路径
        param self.save_json_name 裁剪后的保存的json名字
        param self.cut_dataset_info 裁剪后生成的coco格式的info
        param self.cut_dataset_images 裁剪后的生成的coco格式的images
        param self.cut_dataset_annotations 裁剪后的生成的coco格式的annotations
        param self.cut_dataset_categories 裁剪后的生成的coco格式的categories
        param self.crop_w 裁剪后的图片宽
        param self.crop_h 裁剪后的图片高
        param self.min_crop_w = 600 最小的裁剪框的宽
        param self.min_crop_h = 600 最小的裁剪框的高
        param self.is_flip = is_flip 侧位是否随机翻转
        param self.flip_probability = 0.5 翻转概率
        param self.AP_cut_nums 正位裁剪次数
        param self.LA_cut_nums 侧位裁剪次数
        param self.AP_cut_region 正位裁剪区域
        param self.LA_cut_region 侧位裁剪区域
        param self.cut_image_id 图片id
        param self.cut_ann_id ann id
        param self.save_iou_threshold 裁剪后的框占原框比例大于0.5留下,小于0.5舍弃
        """
        self.images_folder = images_folder
        self.save_path = save_path
        self.save_json_name = save_json_name
        self.cut_dataset_info = dict()
        self.cut_dataset_images = list()
        self.cut_dataset_categories = self.dataset["categories"]
        self.cut_dataset_annotations = list()
        self.cut_dataset = dict()
        self.crop_w = 928
        self.crop_h = 928
        self.min_crop_w = 700
        self.min_crop_h = 700
        self.is_flip = is_flip
        self.flip_probability = 0.5
        self.AP_cut_nums = 2
        self.LA_cut_nums = 2
        self.AP_cut_region = [[11, 6], [6, 1]]
        self.LA_cut_region = [[6, 1], [9, 4]]
        self.cut_image_id = 0
        self.cut_ann_id = 0
        self.save_iou_threshold = 0.5


    def gen_cut_images_and_annotation_file(self):
        self.add_info()
        self.add_categories()
        self.add_images()
        self.save_json()


    def add_info(self):
        self.cut_dataset_info = {"description": "This dataset is labeled as visible to the human eye and labeled: C1-L5,pelvis",
                     "contribute": "Shuai lei",
                     "version": "1.0",
                     "date": datetime.today().strftime('%Y-%m-%d')}
        self.cut_dataset['info'] = self.cut_dataset_info


    def add_categories(self):
        self.cut_dataset["categories"] = self.cut_dataset_categories


    def add_images(self):
        self.cut_images()
        self.cut_dataset["images"] = self.cut_dataset_images
    

    def cut_images(self):
        for image_info in self.dataset["images"]:
            self.cut_image(image_info)


    def cut_image(self, image_info):
        """单张图片裁剪"""
        if image_info["type"] == "AP":
                for AP_num in range(self.AP_cut_nums):
                    # max_category_id与min_category_id为选择的椎体裁剪的大致区域
                    self.cut_image_id += 1
                    cropped_image, cropped_coordinates = self.cut(image_info, max_category_id=self.AP_cut_region[AP_num][0], min_category_id=self.AP_cut_region[AP_num][1])
                    # 如果裁剪的图片框或高小于指定值，则舍弃
                    if cropped_coordinates[2] - cropped_coordinates[0] > self.min_crop_w and cropped_coordinates[3] - cropped_coordinates[1] > self.min_crop_h:
                        cropped_image_file_name = self.get_cropped_image_file_name(image_info, AP_num + 1)
                        self.save_image(cropped_image, cropped_image_file_name)
                        self.add_image(image_info, cropped_image, cropped_image_file_name, cropped_coordinates)
                        self.add_annotations(image_info, cropped_coordinates, 0) # 正位不翻转所以设置为0
        # 侧位,因为侧位不容易区别所以范围会更小
        if image_info["type"] == "LA":
            for LA_num in range(self.LA_cut_nums):
                self.cut_image_id += 1
                cropped_image, cropped_coordinates = self.cut(image_info, max_category_id=self.LA_cut_region[LA_num][0], min_category_id=self.LA_cut_region[LA_num][1])
                # 如果裁剪的图片框或高小于指定值，则舍弃
                if cropped_coordinates[2] - cropped_coordinates[0] > self.min_crop_w and cropped_coordinates[3] - cropped_coordinates[1] > self.min_crop_h:
                    # 随机翻转侧位
                    flip_probability = random.random()
                    cropped_image_file_name = self.get_cropped_image_file_name(image_info, LA_num + 1)
                    if self.is_flip and flip_probability >= self.flip_probability:
                        print(cropped_image_file_name, "flip")
                        flip_cropped_image = cropped_image.transpose(Image.FLIP_LEFT_RIGHT)
                        cropped_image = flip_cropped_image
                    else:
                        print(cropped_image_file_name, "not flip")
                    self.save_image(cropped_image, cropped_image_file_name)
                    self.add_image(image_info, cropped_image, cropped_image_file_name, cropped_coordinates)
                    self.add_annotations(image_info, cropped_coordinates, flip_probability)


    def cut(self, image_info, max_category_id=11, min_category_id=6):
        # 对每张图片裁剪 每张图片裁剪两次
        image = Image.open(os.path.join(self.images_folder, image_info["file_name"]))
        anns = self.imgToAnns[image_info["id"]]
        all_x = 0
        all_y = 0
        num = 0
        for ann in anns:
            if ann["category_id"] <= max_category_id and ann["category_id"] >= min_category_id:
                num += 1
                bbox_center_x, bbox_center_y = self.get_bbox_center(ann)
                all_x += bbox_center_x
                all_y += bbox_center_y
        if num > 0:
            mean_x = all_x / num
            mean_y = all_y / num
            min_x = mean_x-(self.crop_w / 2) if mean_x-(self.crop_w / 2) > 0 else 0 
            min_y = mean_y-(self.crop_h / 2) if mean_y-(self.crop_h / 2) > 0 else 0   
            max_x = mean_x + (self.crop_w / 2) if mean_x+(self.crop_w / 2) < image_info["width"] else image_info["width"]   
            max_y = mean_y + (self.crop_h / 2) if mean_y+(self.crop_h / 2) < image_info["height"] else image_info["height"]   
            cropped_image = image.crop((min_x, min_y, max_x, max_y))
            cropped_coordinates = [min_x, min_y, max_x, max_y]
        else:
            return image, [0, 0, 0, 0]
        return cropped_image, cropped_coordinates
        

    # 生成裁剪后的图片名字
    def get_cropped_image_file_name(self, image_info, cropped_image_id):
        file_name = image_info["file_name"]
        file_name_prefix = file_name.split('.')[0]
        cropped_image_file_name = file_name_prefix + "_" + str(cropped_image_id) + ".jpg"
        return cropped_image_file_name


    # 保存裁剪后的图像
    def save_image(self, cropped_image, cropped_image_file_name):
        cropped_image.save(os.path.join(self.save_path, cropped_image_file_name))


    # 添加裁剪后的图片信息
    def add_image(self, image_info, cropped_image, cropped_image1_file_name, cropped_coordinates):
        cut_image_info = dict()
        cut_image_info["id"] = self.cut_image_id
        cut_image_info["original_image_id"] = image_info["id"]
        cut_image_info["type"] = image_info["type"]
        cut_image_info["L4L6"] = image_info["L4L6"]
        cut_image_info["file_name"] = cropped_image1_file_name
        cut_image_info["width"] = cropped_image.size[0]
        cut_image_info["height"] = cropped_image.size[1]
        cut_image_info["cropped_coordinates"] = cropped_coordinates
        cut_image_info["dataset"] = "BUU Public"
        self.cut_dataset_images.append(cut_image_info)


    # 获取ann的框中心坐标       
    def get_bbox_center(self, ann):
        bbox = ann["bbox"]
        x = bbox[0] + bbox[2] / 2
        y = bbox[1] + bbox[3] / 2
        return x, y
    

    def add_annotations(self, image_info, cropped_coordinates, flip_probability):
        anns = self.imgToAnns[image_info["id"]]
        cut_image_type = image_info["type"]
        for ann in anns:
            iou = self.bbox_iou(ann, cropped_coordinates)
            if iou >= self.save_iou_threshold:
                self.cut_ann_id += 1
                self.add_annotation(cut_image_type, ann, cropped_coordinates, flip_probability)   
        self.cut_dataset["annotations"] = self.cut_dataset_annotations
        
            

    # 添加单个裁剪后的ann
    def add_annotation(self, cut_image_type, ann, cropped_coordinates, flip_probability):
        transformed_bbox_coordinates = self.bbox_coordinates_transformation(ann, cropped_coordinates)
        # 侧位根据概率需要翻转
        if cut_image_type == "LA" and self.is_flip and flip_probability >= self.flip_probability:
            fliped_transformed_bbox_coordinates = self.flip_coordinates(cropped_coordinates, transformed_bbox_coordinates)
        else:
            fliped_transformed_bbox_coordinates = transformed_bbox_coordinates
        cut_ann = {"id": self.cut_ann_id,
                   "image_id": self.cut_image_id,
                   "iscrowd": 0,
                   "category_id": ann["category_id"],
                   "category_name": ann["category_name"],
                   "bbox": fliped_transformed_bbox_coordinates,
                   "area": fliped_transformed_bbox_coordinates[2] * fliped_transformed_bbox_coordinates[3]
                    }
        self.cut_dataset_annotations.append(cut_ann)


    # 判断bbox是否在裁剪区域
    def judgment_bboxes_intersection(self, ann, cropped_coordinates):
        w1, h1 = ann['bbox'][2], ann['bbox'][3]
        w2, h2 = cropped_coordinates[2] - cropped_coordinates[0], cropped_coordinates[3] - cropped_coordinates[1]
        bbox1_center_x, bbox1_center_y = self.get_bbox_center(ann)
        bbox2_center_x, bbox2_center_y = (cropped_coordinates[2] + cropped_coordinates[0]) / 2, (cropped_coordinates[3] + cropped_coordinates[1]) / 2
        bbox_center_distance_x = abs(bbox1_center_x - bbox2_center_x)
        bbox_center_distance_y = abs(bbox1_center_y - bbox2_center_y)
        if bbox_center_distance_x < (w1 + w2)/2.0 and bbox_center_distance_y < (h1 + h2)/2.0:
            return True
        else:
            return False
        

    # 计算iou来决定是否需要留下这个框，iou为相对于类别框的比值
    def bbox_iou(self, ann, cropped_coordinates, eps=1e-9):
        x1, y1, x2, y2 =  ann['bbox'][0],  ann['bbox'][1],  ann['bbox'][0] +  ann['bbox'][2],  ann['bbox'][1] +  ann['bbox'][3]
        x3, y3, x4, y4 = cropped_coordinates[0], cropped_coordinates[1], cropped_coordinates[2], cropped_coordinates[3]
        # 先判断两个框是否相交 
        is_bboxes_intersection = self.judgment_bboxes_intersection(ann, cropped_coordinates)
        if is_bboxes_intersection:
            x_inter1 = max(x1, x3)
            y_inter1 = max(y1, y3)
            x_inter2 = min(x2, x4)
            y_inter2 = min(y2, y4)
            overlap = abs(x_inter2-x_inter1) * abs(y_inter2 - y_inter1)
            area = ann['area'] if 'area' in ann.keys() else abs(ann['bbox'][2] * ann['bbox'][3])
            iou = overlap / area
            return iou
        else:
            return 0


    # 裁剪后坐标变换
    def bbox_coordinates_transformation(self, ann, cropped_coordinates):
        min_x = ann["bbox"][0] - cropped_coordinates[0] if ann["bbox"][0] - cropped_coordinates[0] >= 0 else 0
        min_y = ann["bbox"][1] - cropped_coordinates[1] if ann["bbox"][1] - cropped_coordinates[1] >= 0 else 0
        x1, y1, x2, y2 =  ann['bbox'][0],  ann['bbox'][1],  ann['bbox'][0] +  ann['bbox'][2],  ann['bbox'][1] +  ann['bbox'][3]
        x3, y3, x4, y4 = cropped_coordinates[0], cropped_coordinates[1], cropped_coordinates[2], cropped_coordinates[3]
        x_inter1 = max(x1, x3)
        y_inter1 = max(y1, y3)
        x_inter2 = min(x2, x4)
        y_inter2 = min(y2, y4)
        w = abs(x_inter2 - x_inter1)
        h = abs(y_inter2 - y_inter1)
        transformed_bbox_coordinates = [min_x, min_y, w, h] 
        return transformed_bbox_coordinates
    

    def flip_coordinates(self, cropped_coordinates, transformed_bbox_coordinates):
        center_x = (cropped_coordinates[2] - cropped_coordinates[0]) / 2
        fliped_min_x = 2 * center_x - (transformed_bbox_coordinates[0] + transformed_bbox_coordinates[2])
        fliped_min_y = transformed_bbox_coordinates[1]
        w = transformed_bbox_coordinates[2]
        h = transformed_bbox_coordinates[3]
        fliped_transformed_bbox_coordinates = [fliped_min_x, fliped_min_y, w, h]
        return fliped_transformed_bbox_coordinates

    
    # 保存
    def save_json(self):
        with open(os.path.join(self.save_path, self.save_json_name), "w") as f:
            json.dump(self.cut_dataset, f)
        print("cut images json file save in ", os.path.join(self.save_path, self.save_json_name))