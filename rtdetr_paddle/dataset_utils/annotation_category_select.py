'''
Descripttion: this file in order to from choose category train, instance model need C1-L5,semantic model need vertebrae、 Pelvis、 rib、bone_cement
version: 
Author: ShuaiLei
Date: 2023-11-08 09:59:38
LastEditors: ShuaiLei
LastEditTime: 2024-03-12 10:22:49
'''
import json
import os


class ChooseCatsToCocoJson:
    def __init__(self, annotation_file, choosed_cat_names, save_annotation_file) -> None:
        self.annotation_file = annotation_file
        self.save_annotation_file = save_annotation_file
        self.dataset = self.load_json()
        self.choose_dataset = dict()
        self.choose_dataset_images = list()
        self.choose_dataset_categories = list()
        self.choose_dataset_annotations = list()
        self.choosed_cat_names = choosed_cat_names
        self.choosed_cat_ids = self.get_cat_ids(self.choosed_cat_names)
        self.gen_choosed_coco_json()


    def gen_choosed_coco_json(self):
        self.add_info()
        self.add_categories()
        self.add_images()
        self.add_annotations()
        self.save_json()


    def add_info(self):
        self.choose_dataset["info"] = self.dataset["info"]


    def add_images(self):
        for image_info in self.dataset["images"]:
            file_name = os.path.basename(image_info["file_name"])
            image_info["file_name"] = file_name
            self.choose_dataset_images.append(image_info)
        self.choose_dataset["images"] = self.choose_dataset_images


    def add_categories(self):
        for cat in self.dataset["categories"]:
            if cat["name"]  in self.choosed_cat_names:
                self.choose_dataset_categories.append(cat)
                
        self.choose_dataset["categories"] = self.choose_dataset_categories


    def add_annotations(self):
        for ann in self.dataset["annotations"]:
            if ann["category_id"] in self.choosed_cat_ids:
                self.choose_dataset_annotations.append(ann)

        self.choose_dataset["annotations"] = self.choose_dataset_annotations


    def get_cat_ids(self, cat_names):
        cat_ids = []

        for cat in self.dataset["categories"]:
            if cat["name"] in cat_names:
                cat_ids.append(cat["id"])

        return cat_ids

    def load_json(self):
        
        with open(self.annotation_file, "r") as f:
            dataset = json.load(f)

        return dataset
    

    def save_json(self):
        with open(self.save_annotation_file, "w") as f:
            json.dump(self.choose_dataset, f)
        print(self.save_annotation_file, "save completed!")


if __name__ == "__main__":
    # LY 20231105
    instance = ChooseCatsToCocoJson("datasets/LY20231105/annotations/result.json", 
                                    ["L5", "L4", "L3", "L2", "L1",
                                    "T12", "T11", "T10", "T9", "T8", "T7", "T6","T5", "T4", "T3", "T2", "T1", 
                                    "C7", "C6", "C5", "C4","C3", "C2", "C1"], 
                                    "datasets/LY20231105/annotations/instance.json")
    
    semantic = ChooseCatsToCocoJson("datasets/LY20231105/annotations/result.json", 
                                       ["Pelvis", "L5", "L4", "L3", "L2", "L1",
                                        "T12", "T11", "T10", "T9", "T8", "T7", "T6","T5", "T4", "T3", "T2", "T1", 
                                        "C7", "C6", "C5", "C4","C3", "C2", "C1", "rib", "bone_cement"], 
                                        "datasets/LY20231105/annotations/instance_all.json")
    
    # XJT 
    # instance = ChooseCatsToCocoJson("datasets/XJT20231101/annotations/result.json", 
    #                                 ["L5", "L4", "L3", "L2", "L1",
    #                                 "T12", "T11", "T10", "T9", "T8", "T7", "T6","T5", "T4", "T3", "T2", "T1", 
    #                                 "C7", "C6", "C5", "C4","C3", "C2", "C1"], 
    #                                 "datasets/XJT20231101/annotations/instance.json")
    
    # semantic = ChooseCatsToCocoJson("datasets/XJT20231101/annotations/result.json", 
    #                                    ["Pelvis", "L5", "L4", "L3", "L2", "L1",
    #                                     "T12", "T11", "T10", "T9", "T8", "T7", "T6","T5", "T4", "T3", "T2", "T1", 
    #                                     "C7", "C6", "C5", "C4","C3", "C2", "C1", "rib", "bone_cement"], 
    #                                     "datasets/XJT20231101/annotations/instance_all.json")
    
    # TD 20240117
    # instance = ChooseCatsToCocoJson("datasets/TD20240117/annotations/result.json", 
    #                                 ["L5", "L4", "L3", "L2", "L1",
    #                                 "T12", "T11", "T10", "T9", "T8", "T7", "T6","T5", "T4", "T3", "T2", "T1", 
    #                                 "C7", "C6", "C5", "C4","C3", "C2", "C1"], 
    #                                 "datasets/TD20240117/annotations/instance.json")
    
    # semantic = ChooseCatsToCocoJson("datasets/TD20240117/annotations/result.json", 
    #                                    ["Pelvis", "L5", "L4", "L3", "L2", "L1",
    #                                     "T12", "T11", "T10", "T9", "T8", "T7", "T6","T5", "T4", "T3", "T2", "T1", 
    #                                     "C7", "C6", "C5", "C4","C3", "C2", "C1", "rib", "bone_cement"], 
    #                                     "datasets/TD20240117/annotations/instance_all.json")

