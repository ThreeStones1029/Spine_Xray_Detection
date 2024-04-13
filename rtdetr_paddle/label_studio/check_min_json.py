'''
Descripttion: 
version: 
Author: ShuaiLei
Date: 2023-11-14 16:44:57
LastEditors: ShuaiLei
LastEditTime: 2023-11-14 16:59:14
'''
import json


class Check_Min_Json:
    def __init__(self, annotation_file) -> None:
        self.annotation_file = annotation_file
        with open(self.annotation_file, "r") as f:
            self.dataset = json.load(f)


    def load_anns(self):
        i = 0
        for image_info in self.dataset:
            if "sentiment" in image_info.keys():
                if image_info["sentiment"] == 'True': # "sentiment"代表标注难易
                    i += 1
                    print(image_info["id"])
        print(i)


if __name__ == "__main__":
    Check_Min_Json("datasets/BUU/project-1-at-2023-11-14-08-34-c711374c.json").load_anns()