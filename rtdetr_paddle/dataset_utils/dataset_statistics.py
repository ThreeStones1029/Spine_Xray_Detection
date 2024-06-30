'''
Descripttion: this file will used to Analyze the dataset composition.
version: 
Author: ShuaiLei
Date: 2023-11-11 10:56:44
LastEditors: ShuaiLei
LastEditTime: 2024-06-28 07:36:10
'''
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "rtdetr_paddle"))
from dataset_utils.blend_annotation_file import Gen_Blend_X_Ray_Dataset
import matplotlib.pyplot as plt


class DatasetStatistics:
    def __init__(self, annotation_file, save_json_file):
        self.annotation_file = annotation_file
        self.save_json_file = save_json_file
        self.root_folder = os.path.dirname(self.annotation_file)
        self.dataset = self.load_json()
        self.dataset_info_statistics = dict()
        self.overall_information = dict()
        self.categories_number_information = dict()
        self.statistic()

    # 统计分析数据集
    def statistic(self):
        self.overall_information_statistic()
        self.categories_statistic()
        self.print_information()
        self.plot_images()
        self.save_information()

    # 整体信息统计
    def overall_information_statistic(self):
        self.overall_information["images_num"] = len(self.dataset["images"])
        self.overall_information["AP_images_num"] = 0
        self.overall_information["LA_images_num"] = 0
        self.overall_information["annnotations_num"] = len(self.dataset["annotations"])
        self.overall_information["categories_num"] = len(self.dataset["categories"])
        # 统计图片大小
        self.overall_information["image_width_information"] = {}
        self.overall_information["image_height_information"] = {}
        for image in self.dataset["images"]:
            if image["width"] not in self.overall_information["image_width_information"].keys():
                self.overall_information["image_width_information"][image["width"]] = 1
            else:
                self.overall_information["image_width_information"][image["width"]] += 1
            if image["height"] not in self.overall_information["image_height_information"].keys():
                self.overall_information["image_height_information"][image["height"]] = 1
            else:
                self.overall_information["image_height_information"][image["height"]] += 1
            if "type" in image.keys():
                if image["type"] == "AP":
                    self.overall_information["AP_images_num"] += 1
                if image["type"] == "LA":
                    self.overall_information["LA_images_num"] += 1
        # 求平均图片大小
        all_width = 0
        all_height = 0
        for width, number in self.overall_information["image_width_information"].items():
            all_width += width*number
        for height, number in self.overall_information["image_height_information"].items():
            all_height += height *number
        self.overall_information["mean_width"] = all_width / self.overall_information["images_num"]
        self.overall_information["mean_height"] = all_height / self.overall_information["images_num"]
        self.dataset_info_statistics["Overall information"] = self.overall_information

    # 类别统计分析
    def categories_statistic(self):
        self.init_categories_number()
        for ann in self.dataset["annotations"]:
            self.categories_number_information[ann["category_name"]]["number"] += 1
        self.compute_proportion_of_categries()
        self.dataset_info_statistics["categories_information"] = self.categories_number_information


    # 初始化类别数量
    def init_categories_number(self):
        for cat in self.dataset["categories"]:
            cat_info = dict() # 注意字典初始化若在循环外面会统计所有类别数量，因为修改每个类别数量时修改的是同一个字典
            cat_info["number"] = 0
            self.categories_number_information[cat["name"]] = cat_info


    # 计算类别数量比例
    def compute_proportion_of_categries(self):
        for cat in self.dataset["categories"]:
            self.categories_number_information[cat["name"]]["ratio"] = self.categories_number_information[cat["name"]]["number"] / self.overall_information["annnotations_num"]*100


    def print_information(self):
        for key, value in self.dataset_info_statistics.items():
            print(key)
            print(value)


    def plot_images(self):
        self.plot_bar("image_width_information", os.path.join(self.root_folder, "images_width_distribution.png"), value_condition=20)
        self.plot_bar("image_height_information", os.path.join(self.root_folder, "images_height_distribution.png"), value_condition=20)

 
    def plot_bar(self, information_name, save_path, value_condition=0):
        data = sorted(self.overall_information[information_name].items(), key= lambda x: x[0])
        x = [int(item[0]) for item in data if item[1] >= value_condition]
        y = [int(item[1]) for item in data if item[1] >= value_condition]
        print(len(x))
        plt.bar(x, y, color="blue")
        plt.title(information_name)
        xlabel = information_name.split("_")[1]
        plt.xlabel(xlabel)
        plt.ylabel("number")
        plt.savefig(save_path)
        plt.close()


    # 保存信息
    def save_information(self):
        with open(self.save_json_file, "w") as f:
            json.dump(self.dataset_info_statistics, f)

    # 加载数据集文件
    def load_json(self):
        if type(self.annotation_file) == str:
            with open(self.annotation_file, "r") as f:
                dataset = json.load(f)
        return dataset


if __name__ == "__main__":
    # Gen_Blend_X_Ray_Dataset(annotation_file1="datasets/all_intraoperative_xray/annotations/train_instance.json",
    #                         annotation_file2="datasets/all_intraoperative_xray/annotations/val_test_instance.json",
    #                         blend_annotation_file="datasets/all_intraoperative_xray/annotations/train_and_val_test_instance.json",
    #                         blend_start_image_id=0,
    #                         blend_start_ann_id=0).blend_two_instance_dataset()
    
    # DatasetStatistics(annotation_file="datasets/all_intraoperative_xray/annotations/train_and_val_test_instance.json",
    #                   save_json_file="datasets/all_intraoperative_xray/annotations/train_and_val_test_instance_statistic.json")

    # miccai xray143标注信息分布
    # DatasetStatistics(annotation_file="datasets/miccai/xray/annotations/xray_instance.json",
    #                   save_json_file="datasets/miccai/xray/annotations/xray_instance_statistic.json")

    # miccai BUU5800数据集标注信息分布
    # DatasetStatistics(annotation_file="datasets/miccai/BUU/annotations/buu_5800.json",
    #                   save_json_file="datasets/miccai/BUU/annotations/buu_5800_statistic.json")

    # xray20240119数据集标注信息分布
    DatasetStatistics(annotation_file="datasets/xray20240119/annotations/train_instance.json",
                      save_json_file="datasets/xray20240119/annotations/train_instance_statistics.json")