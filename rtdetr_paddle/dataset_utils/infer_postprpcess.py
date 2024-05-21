'''
Description: 在推理前需要对于具体标签模型结果和不具体标签推理前进行后处理
version: 1.0
Author: ThreeStones1029 221620010039@hhu.edu.cn
Date: 2023-10-02 14:49:08
LastEditors: ShuaiLei
LastEditTime: 2024-03-03 16:28:50
'''
import os
import sys
import json
import numpy as np
from dataset_utils.bbox_utils import bbox_iou


# 删除相交iou大于某个阈值的框，留下其中置信度较高的
def filter_bbox_intersection(single_img_results, iou_threshold):
    postprocess_single_img_results = []
    while(single_img_results):
        to_be_selected_anns = []
        # 顶部框出栈，且加入待选择的框列表
        cur_ann = single_img_results.pop()
        to_be_selected_anns.append(cur_ann)
        to_be_deleted_anns = []
        # 计算当前框与剩下的框的iou,若iou大于某个阈值，则加入待选择框,同时加入待删除框
        for other_ann in single_img_results:
            iou = bbox_iou(cur_ann, other_ann)
            if iou > iou_threshold:
                # 记录大于某个阈值的框
                to_be_deleted_anns.append(other_ann)
                to_be_selected_anns.append(other_ann)
        # 找到其中置信度最大的一个
        max_confidence = 0
        for ann in to_be_selected_anns:
            if ann['score'] > max_confidence:
                max_confidence = ann['score']
                max_confidence_ann = ann
        # 最大置信度的加入最终结果
        postprocess_single_img_results.append(max_confidence_ann)
        # 删除已经计算了iou且，iou大于threshold的ann
        for ann in to_be_deleted_anns:
            single_img_results.remove(ann)
    return postprocess_single_img_results



if __name__ == "__main__":
    single_bbox_list = [{'image_id': 38, 
                         'category_id': 10, 
                         'bbox': [248.3925018310547, 13.044088363647461, 280.0681915283203, 143.4751377105713], 
                         'score': 0.8588865399360657, 'file_name': 'qcx_1.png', 'category_name': 'L2'}, 
                         {'image_id': 38, 
                          'category_id': 11, 
                          'bbox': [258.971435546875, 199.298095703125, 287.38165283203125, 168.29891967773438], 
                          'score': 0.9624585509300232, 'file_name': 'qcx_1.png', 'category_name': 'L3'}] 
    
    multiple_bboxes_list = [[{'image_id': 38, 
                              'category_id': 13, 
                              'bbox': [72.48463439941406, 529.6393432617188, 261.7639617919922, 159.89013671875], 
                              'score': 0.6692593693733215, 'file_name': 'qcx_1.png', 'category_name': 'L5'}, 
                              {'image_id': 38, 
                               'category_id': 13, 
                               'bbox': [244.7266082763672, 386.43609619140625, 299.63026428222656, 198.52398681640625], 
                               'score': 0.6031084060668945, 'file_name': 'qcx_1.png', 'category_name': 'L5'}]]
    