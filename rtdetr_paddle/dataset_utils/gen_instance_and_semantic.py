'''
Descripttion: 
version: 
Author: ShuaiLei
Date: 2023-11-09 12:12:40
LastEditors: ShuaiLei
LastEditTime: 2024-03-12 10:23:25
'''
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "rtdetr_paddle"))
from dataset_utils.annotation_category_select import ChooseCatsToCocoJson
from dataset_utils.conver_label import instance_label_to_semantic_label
from dataset_utils.conver_label import ModifyCatId
from vis_utils.vis_coco import VisCoCo


# 从label-studio导出后的result.json挑选需要的类别
# 具体标签模型
instance = ChooseCatsToCocoJson("datasets/XJT20231101/annotations/result.json", 
                                ["L5", "L4", "L3", "L2", "L1",
                                "T12", "T11", "T10", "T9", "T8", "T7", "T6","T5", "T4", "T3", "T2", "T1", 
                                "C7", "C6", "C5", "C4","C3", "C2", "C1"], 
                                "datasets/XJT20231101/annotations/instance.json")
# 不具体标签模型  
semantic = ChooseCatsToCocoJson("datasets/XJT20231101/annotations/result.json", 
                                    ["Pelvis", "L5", "L4", "L3", "L2", "L1",
                                    "T12", "T11", "T10", "T9", "T8", "T7", "T6","T5", "T4", "T3", "T2", "T1", 
                                    "C7", "C6", "C5", "C4","C3", "C2", "C1", "rib", "bone_cement"], 
                                    "datasets/XJT20231101/annotations/instance_all.json")

instance_label_to_semantic_label("datasets/XJT20231101/annotations/instance_all.json",
                                    "datasets/XJT20231101/annotations/semantic.json")

# 修改id
ModifyCatId(annotation_file="datasets/XJT20231101/annotations/instance.json",
            modify_annotation_file="datasets/XJT20231101/annotations/instance.json").modify_instance()

ModifyCatId(annotation_file="datasets/XJT20231101/annotations/semantic.json",
            modify_annotation_file="datasets/XJT20231101/annotations/semantic.json").modify_semantic()

# 可视化
vis = VisCoCo(annotation_file="datasets/XJT20231101/annotations/instance.json", 
              images_folder="datasets/XJT20231101/images", 
              save_images_folder="datasets/XJT20231101/vis_instance").visualize_images()
    
vis = VisCoCo(annotation_file="datasets/XJT20231101/annotations/semantic.json", 
              images_folder="datasets/XJT20231101/images", 
              save_images_folder="datasets/XJT20231101/vis_semantic").visualize_images()