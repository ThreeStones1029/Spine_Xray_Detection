'''
Descripttion: 
version: 
Author: ShuaiLei
Date: 2023-10-27 11:18:25
LastEditors: ShuaiLei
LastEditTime: 2023-10-27 11:18:30
'''
import json
import os


def modify_bbox_json_imid(truth_json_path, imid2path, output_dir):
    path2imid = {value:key for key,value in imid2path.items()}
    # 得到真实图片名字对应的id
    file_name_to_image_id = {}
    with open(truth_json_path, 'r') as f:
        truth_json_data = json.load(f)
    for img in truth_json_data['images']:
        file_name_to_image_id[img['file_name']] = img['id']
    
    # 记录目前id与真实json文件里面的图片id
    imid2truth_im_id = {}
    for imid, path in imid2path.items():
        filename = os.path.basename(path)
        # 若当前图片名字存在真实标注文件图片名字中，则修改，否则不变
        if filename in file_name_to_image_id.keys():
            truth_im_id = file_name_to_image_id[filename]
        else:
            truth_im_id = imid
        imid2truth_im_id[path2imid[path]] = truth_im_id

    # 修改im_id
    with open(os.path.join(output_dir, 'bbox.json'), 'r') as f_read:
        predict_bbox_data = json.load(f_read)
    for ann in predict_bbox_data:
        ann['image_id'] = imid2truth_im_id[ann['image_id']]
    # 保存
    with open(os.path.join(output_dir, 'bbox.json'), 'w', encoding='utf-8') as f_write:
        json.dump(predict_bbox_data, f_write)