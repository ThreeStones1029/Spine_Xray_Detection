'''
Descripttion: 
version: 
Author: ShuaiLei
Date: 2023-08-30 10:11:03
LastEditors: ShuaiLei
LastEditTime: 2024-03-12 10:28:01
'''
import json
import os


def instance_label_to_semantic_label(instance_json_path, semantic_json_path):
    if not os.path.exists(instance_json_path):
        raise ValueError("json path `{}` does not exists. ".format(instance_json_path))

    with open(instance_json_path, "r") as f:
        dataset = json.load(f)

    dataset['info'] = {
                        "info": {
                            "description": "This dataset is labeled as visible to the human eye and labeled: vertebrae,pelvis",
                            "contribute": "Shuai lei",
                            "version": "1.0",
                            "date": "2023-10-26"
                        }}
    
    for ann in dataset["annotations"]:
        if ann["category_name"] != "Pelvis":
            ann["category_id"] = 0
            ann["category_name"] = "vertebrae"

    dataset['categories'] = [{
                            "id": 0,
                            "name": "vertebrae",
                            "supercategory": "bone"
                            },
                            {
                            "id": 1,
                            "name": "Peivis",
                            "supercategory": "bone"
                            },
                            ]


    with open(semantic_json_path, "w", encoding='utf-8') as w:
        json.dump(dataset, w)
    print("转换成功")


if __name__ == "__main__":
    instance_label_to_semantic_label("datasets/BUU/sample/annotations/bbox_test.json",
                                       "datasets/BUU/sample/annotations/bbox_test_semantic.json")