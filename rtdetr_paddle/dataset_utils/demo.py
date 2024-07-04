'''
Description: 
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-07-01 07:08:52
LastEditors: ShuaiLei
LastEditTime: 2024-07-01 07:16:15
'''
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "rtdetr_paddle"))
from json_process import load_json_file, save_json_file


def modify_categories_id(json_file_path):
    """
    修改类别id,同时添加类别名字,由从0开始变成从1开始
    """
    dataset = load_json_file(json_file_path)
    catid2catname = {category["id"]: category["name"] for category in dataset["categories"]}
    for ann in dataset["annotations"]:
        ann["category_name"] = catid2catname[ann["category_id"]]
        ann["category_id"] += 1
    for category in dataset["categories"]:
        category["id"] += 1
    save_json_file(dataset, json_file_path)


if __name__ == "__main__":
    modify_categories_id("datasets/LA_xray_fracture/annotations/LA_xray_fracture.json")