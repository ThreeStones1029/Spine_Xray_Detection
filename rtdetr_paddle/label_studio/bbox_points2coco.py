'''
Descripttion: "this py file will be used to converse the json file which is exported to coco format.
version: 1.0
Author: ShuaiLei
Date: 2023-10-24 14:21:48
LastEditors: ShuaiLei
LastEditTime: 2023-11-07 10:15:27


BUU datasets annotations format:
    AP
        keypoints:
            C1——L5 4 points
        bbox:
            pelvis 1 bbox
    LA
        keypoints:
            C1——L5 4 points
            pelvis 3 points

CoCo annotations format:
    AP、LA
        bbox:
            C1——L5、pelvis total 24 bboxes
'''
import json
import os 
from datetime import datetime
from collections import defaultdict


class Label_studio2coco():
    def __init__(self, annotation_file=None, save_path=None):
        self.annotations_file = annotation_file
        self.save_path = save_path
        if annotation_file != None:
            print('loading annotations into memory...')
            with open(self.annotations_file, "r") as f:
                self.label_studio_annotations = json.load(f)
            self.coco_annotations = {}
            self.dataset,self.anns,self.cats,self.imgs, self.info = dict(),dict(),dict(),dict(), dict()
            self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
            self.conver_coco()
            print("conver label studio json file to coco format complete!")
        else:
            print("the file path is none")


    def conver_coco(self):
        self.add_info()
        self.add_images()
        self.add_categories()
        self.add_annotations()
        self.save_dataset()

    def add_info(self):
        self.info = {"description": "This dataset is labeled as visible to the human eye and labeled: C1-L5,pelvis",
                     "contribute": "Shuai lei",
                     "version": "1.0",
                     "date": datetime.today().strftime('%Y-%m-%d')}
        self.dataset['info'] = self.info


    def add_categories(self):
        categories = []
        cats = {}
        categories.append({"id": 1,
                           "name": "Pelvis", 
                           "supercategory": "vertebrae"})
        for i in range(2, 7):
            categories.append({"id": i,
                               "name": "L" + str(7-i),
                               "supercategory": "vertebrae"}) 
        for i in range(7, 19):
            categories.append({"id": i,
                               "name": "T" + str(19-i),
                               "supercategory": "vertebrae"}) 
        for i in range(19, 26):
            categories.append({"id": i,
                               "name": "C" + str(26-i),
                               "supercategory": "vertebrae"})  
        self.dataset['categories'] = categories

        for cat in self.dataset['categories']:
            cats[cat['id']] = cat
        self.cats = cats


    def add_images(self):
        images = []
        imgs = {}
        for img_info in self.label_studio_annotations:
            if img_info['annotator'] != 3: # 排除原始BUU数据自带的标注
                img = {}
                img['id'] = img_info['id']
                img['type'] = img_info['type']
                img['L4L6'] = img_info['L4L6']
                img['file_name'] = os.path.basename(img_info['img'])
                img['width'] = img_info['vertebrae-point'][0]['original_width']
                img['height'] = img_info['vertebrae-point'][0]['original_height']
                images.append(img)
        self.dataset['images'] = images

        for img in self.dataset['images']:
            imgs[img['id']] = img
        self.imgs = imgs


    def add_annotations(self):
        annotations = []
        anns = {}
        ann_id = 1
        # 对于所有图片
        for img_info in self.label_studio_annotations:
            if img_info['annotator'] != 3: # 排除原始BUU数据自带的标注
                cat_name2keypoints = defaultdict(list)
                cat_name2bboxes = defaultdict(list)
                # 将同一个类别的点加入到同一个列表
                if 'vertebrae-point' in img_info.keys():
                    for point in img_info['vertebrae-point']:
                        if point['keypointlabels'][0] != "Pelvis": # 侧位骨盆有标注框，就不需要点了
                            cat_name2keypoints[point['keypointlabels'][0]].append(point)
                # 将框加入字典列表
                if 'bbox' in img_info.keys():
                    for bbox in img_info['bbox']:
                        cat_name2bboxes[bbox['rectanglelabels'][0]].append(bbox)

                cat_name2id = {category['name']: category['id'] for category in self.dataset['categories']}

                # 将单张图片 先加入标志点围成的框
                for cat_name, points in cat_name2keypoints.items():
                    ann = {}
                    ann['id'] = ann_id
                    ann['image_id'] = img_info['id']
                    ann['iscrowd'] = 0
                    ann['category_id'] = cat_name2id[cat_name]
                    ann['category_name'] = cat_name
                    x = []
                    y = []
                    for point in points:
                        x.append(point['x'] * point['original_width'] / 100)
                        y.append(point['y'] * point['original_height'] / 100)
                    min_x = min(x)
                    min_y = min(y)
                    max_x = max(x)
                    max_y = max(y)
                    bbox_width = max_x - min_x
                    bbox_height = max_y - min_y
                    bbox = [min_x, min_y, bbox_width, bbox_height]
                    ann['bbox'] = bbox
                    ann['area'] = bbox_height * bbox_width
                    ann_id += 1
                    annotations.append(ann)

                # 加入标注的框信息
                for cat_name, bbox in cat_name2bboxes.items():
                    ann = {}
                    ann['id'] = ann_id
                    ann['image_id'] = img_info['id']
                    ann['iscrowd'] = 0
                    ann['category_id'] = cat_name2id[cat_name]
                    ann['category_name'] = cat_name
                    ann['bbox'] = [bbox[0]['x'] * bbox[0]['original_width'] / 100, 
                                   bbox[0]['y'] * bbox[0]['original_height'] /100, 
                                   bbox[0]['width'] * bbox[0]['original_width'] / 100, 
                                   bbox[0]['height'] * bbox[0]['original_height'] / 100]
                    ann['area'] = ann['bbox'][2] * ann['bbox'][3]
                    ann_id += 1
                    annotations.append(ann)
        self.dataset['annotations'] = annotations

        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        for ann in self.dataset['annotations']:
            imgToAnns[ann['image_id']].append(ann)
            catToImgs[ann['category_id']].append(ann['image_id'])
            anns[ann['id']] = ann
        self.imgToAnns = imgToAnns
        self.anns = anns


    def save_dataset(self):
        with open(self.save_path, "w") as w:
            json.dump(self.dataset, w)


if __name__ == "__main__":
    conversion = Label_studio2coco("XJT_tools/project-9-at-2023-11-04-15-47-5e327061.json", "XJT_tools/BUU_cut.json")

    # vis = VisCoCo("datasets/BUU/sample/annotations/bbox_val.json", "datasets/BUU/sample/val")
    # vis.visualize_img(8073, "datasets/BUU/vis/8072.png")