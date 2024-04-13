'''
Descripttion: 
version: 
Author: ShuaiLei
Date: 2023-10-25 17:20:06
LastEditors: ShuaiLei
LastEditTime: 2023-10-25 17:25:09
'''
import os
import json
from datetime import datetime
import shutil
import numpy as np


def random_split_coco_dataset(dataset_root, json_name):
    """
    随机划分json文件,并划分好相应的数据集
    """
    
    # 数据集路径
    
    images_folder = os.path.join(dataset_root, "images")
    annotations_path = os.path.join(dataset_root,"annotations_sample" ,json_name)
    
    # 输出路径
    output_root = os.path.join(dataset_root, "split_dataset")
    os.makedirs(output_root, exist_ok=True)
    
    # 读取annotations.json文件
    with open(annotations_path, "r") as f:
        annotations_data = json.load(f)
    
    # 提取images, annotations, categories
    info = annotations_data["info"]
    info['date'] = datetime.today().strftime('%Y-%m-%d')
    images = annotations_data["images"]
    annotations = annotations_data["annotations"]
    categories = annotations_data["categories"]
    
    # 随机打乱数据
    np.random.shuffle(images)
    
    # 训练集，验证集，测试集比例 数据集较少时不划分测试集
    # train_ratio, val_ratio, test_ratio = 0.8, 0.2, 0
    train_ratio, val_ratio = 0.75, 0.25
    
    # 计算训练集，验证集，测试集的大小
    num_images = len(images)
    num_train = int(num_images * train_ratio)
    num_val = int(num_images * val_ratio)
    
    # 划分数据集
    train_images = images[:num_train]
    val_images = images[num_train:]
    # val_images = images[num_train:num_train + num_val]
    # test_images = images[num_train + num_val:]
    
    # 分别为训练集、验证集和测试集创建子文件夹
    train_folder = os.path.join(output_root, "train")
    val_folder = os.path.join(output_root, "val")
    # test_folder = os.path.join(output_root, "test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    # os.makedirs(test_folder, exist_ok=True)
    
    # 将图片文件复制到相应的子文件夹
    for img in train_images:
        shutil.copy(os.path.join(images_folder, img["file_name"]), os.path.join(train_folder, img["file_name"]))
    
    for img in val_images:
        shutil.copy(os.path.join(images_folder, img["file_name"]), os.path.join(val_folder, img["file_name"]))
    
    # for img in test_images:
    #     shutil.copy(os.path.join(images_folder, img["file_name"]), os.path.join(test_folder, img["file_name"]))
    
    # 根据图片id分配annotations
    def filter_annotations(annotations, image_ids):
        return [ann for ann in annotations if ann["image_id"] in image_ids]
    
    train_ann = filter_annotations(annotations, [img["id"] for img in train_images])
    val_ann = filter_annotations(annotations, [img["id"] for img in val_images])
    # test_ann = filter_annotations(annotations, [img["id"] for img in test_images])
    
    # 生成train.json, val.json, test.json
    train_json = {"info": info, "images": train_images, "annotations": train_ann, "categories": categories}
    val_json = {"info": info, "images": val_images, "annotations": val_ann, "categories": categories}
    # test_json = {"info": info, "images": test_images, "annotations": test_ann, "categories": categories}
    
    with open(os.path.join(output_root,"bbox_train.json"), "w") as f:
        json.dump(train_json, f)
    
    with open(os.path.join(output_root, "bbox_val.json"), "w") as f:
        json.dump(val_json, f)
    
    # with open(os.path.join(output_root, "bbox_test.json"), "w") as f:
    #     json.dump(test_json, f)
    
    print("数据集划分完成！")


if __name__ == "__main__":
    random_split_coco_dataset("datasets/BUU","sample.json")