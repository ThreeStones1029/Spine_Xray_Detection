'''
Descripttion: 
version: 
Author: ShuaiLei
Date: 2023-11-10 17:15:16
LastEditors: ShuaiLei
LastEditTime: 2023-11-10 20:08:33
'''
import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "rtdetr_paddle"))
from label_studio.min_json2coco import Min_json2coco
from dataset_utils.image_cut import Cut_Image
from dataset_utils.choose_images_from_annotation_file import choose_images_from_coco_json
from vis_utils.vis_coco import VisCoCo


def main():
    # 转换格式
    print("label studio min json format conver to coco json")
    conversion = Min_json2coco(annotation_file="datasets/BUU/sample514_cut/annotations/project-9-at-2023-11-04-15-47-5e327061.json", 
                               choose_ids="all",
                               save_path="datasets/BUU/sample514_cut/annotations/buu514_coco.json")

    # 挑选标注的图片
    print("gen dataset images not cut")
    choose_images_from_coco_json(coco_json_file_path="datasets/BUU/sample514_cut/annotations/buu514_coco.json", 
                                 images_folder="/opt/XRay_Data/BUU-LSPINEv1/AP", 
                                 choosed_images_folder="datasets/BUU/sample514_cut/images")
    

    choose_images_from_coco_json(coco_json_file_path="datasets/BUU/sample514_cut/annotations/buu514_coco.json", 
                                 images_folder="/opt/XRay_Data/BUU-LSPINEv1/LA", 
                                 choosed_images_folder="datasets/BUU/sample514_cut/images")
    
    # 生成裁剪图片
    print("cuting...")
    cut = Cut_Image(annotation_file="datasets/BUU/sample514_cut/annotations/buu514_coco.json",
                    images_folder="datasets/BUU/sample514_cut/images",
                    is_flip=True,
                    save_path="datasets/BUU/sample514_cut/cut_images",
                    save_json_name = "buu514_coco_cut.json")
    
    cut.gen_cut_images_and_annotation_file()
    print("images cut complete!")

    # 可视化裁剪后的图片
    VisCoCo(annotation_file="datasets/BUU/sample514_cut/cut_images/buu514_coco_cut.json", 
            images_folder="datasets/BUU/sample514_cut/cut_images", 
            save_images_folder="datasets/BUU/sample514_cut/vis_cut_images").visualize_images()


if __name__ == "__main__":
    main()