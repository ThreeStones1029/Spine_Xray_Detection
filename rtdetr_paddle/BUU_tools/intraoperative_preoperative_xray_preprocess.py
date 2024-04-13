'''
Descripttion: this file is used to blend x_ray to generate blended dataset(术中和术前)
version: 
Author: ShuaiLei
Date: 2023-11-06 11:31:56
LastEditors: ShuaiLei
LastEditTime: 2024-03-12 10:22:10
'''
import sys
import os
import json
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "rtdetr_paddle"))
from vis_utils.vis_coco import VisCoCo
from dataset_utils.blend_annotation_file import Gen_Blend_X_Ray_Dataset
from dataset_utils.image_cut import Cut_Image


def cut_buu_images():
    cut = Cut_Image(annotation_file="XJT/json_file/BUU_coco.json",
                    images_folder="XJT/images/choosed_BUU_images",
                    save_path="XJT/images/cut_BUU_images",
                    save_json_name = "preoperative.json")
    cut.gen_cut_images_and_annotation_file()
    VisCoCo("XJT/images/cut_BUU_images/preoperative.json", "XJT/images/cut_BUU_images", "XJT/images/vis_cut_BUU_images").visualize_images()


def blend_intraoperative_and_preoperative():
    train = Gen_Blend_X_Ray_Dataset(annotation_file1="XJT/images/cut_BUU_images/preoperative.json", 
                                    annotation_file2="datasets/Final_instance_Label_Modify/annotations/bbox_train.json", 
                                    blend_annotation_file="datasets/Blend_Pre_Intra/annotations/train.json", 
                                    blend_start_image_id=0, 
                                    blend_start_ann_id=0)
    # 统计混合后的训练集图片与ann数量
    train_images_num = len(train.blend_dataset_images)
    train_annotations_num = len(train.blend_dataset_annotations)

    Gen_Blend_X_Ray_Dataset(annotation_file1="datasets/Final_instance_Label_Modify/annotations/bbox_test.json", 
                            annotation_file2="datasets/Final_instance_Label_Modify/annotations/bbox_val.json",
                            blend_annotation_file="datasets/Blend_Pre_Intra/annotations/val_test.json", 
                            blend_start_image_id=train_images_num, 
                            blend_start_ann_id=train_annotations_num)
    
    vis = VisCoCo("datasets/Blend_Pre_Intra/annotations/train.json", "datasets/Blend_Pre_Intra/train", "datasets/Blend_Pre_Intra/vis_train")
    vis.visualize_images()


def main():
    cut_buu_images()
    blend_intraoperative_and_preoperative()


if __name__ == "__main__":
    main()