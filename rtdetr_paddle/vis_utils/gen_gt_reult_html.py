'''
Descripttion: this file in order to show GT and Result bboxes in a html.
version: 
Author: ShuaiLei
Date: 2023-10-26 10:45:40
LastEditors: ShuaiLei
LastEditTime: 2024-03-12 10:28:22
'''
from dominate import document
from dominate.tags import div, img, h3, span
import os
from vis_utils.vis_coco import VisCoCo
from glob import glob
import multiprocessing


class HTML:
    def __init__(self):
        pass


# 可视化真实标注
def gen_gt_images_annotations(vis, file_path, gt_vis_save_folder):
    file = os.path.basename(file_path)
    img_id = vis.from_file_name2img_id(file)
    save_img_path = gt_vis_save_folder + file
    vis.visualize_img(img_id, save_img_path)
    

def gen_images_data(annotations_file, infer_folder, gt_vis_save_folder, pre_vis_folder):
    vis = VisCoCo(annotations_file, infer_folder)
    files_path = glob(os.path.join(infer_folder, '*.jpg'))
    pool = multiprocessing.Pool(8) # 创建8个进程，提高代码处理效率
    pool.starmap(gen_gt_images_annotations, [(vis, file_path, gt_vis_save_folder) for file_path in sorted(files_path)])

    # 将图片路径加入到列表
    images_data = []
    for root, dirs, files in os.walk(infer_folder):
        for file in files:
            gt_vis_img_path = os.path.join(gt_vis_save_folder, file)
            instance_pre_vis_path = os.path.join(pre_vis_folder, "instance", file)
            semantic_pre_vis_path = os.path.join(pre_vis_folder, "semantic", file)
            images_data.append({"gt": gt_vis_img_path, "instance_pre":instance_pre_vis_path, "semantic_pre":semantic_pre_vis_path})

    return images_data


def gen_gt_result_html(images_data):
    doc = document(title="Gt and Result")
    with doc:
        with div(style="display: flex"):
            with div(style="flex: 1; text-align:center"):
                h3("GT")
                for i, data in enumerate(images_data, start=1):
                    img(src=data['gt'], style="max-width: 90%")
                    with h3():
                        span(f"Image {i}", ":", style="color: red")
                        span(os.path.basename(data['gt']))

            with div(style="flex: 1; text-align:center"):
                h3("instance Result")
                for i, data in enumerate(images_data, start=1):
                    img(src=data['instance_pre'], style="max-width: 90%") 
                    with h3():
                        span(f"Image {i}", ":", style="color: red")
                        span(os.path.basename(data['instance_pre']))   

            with div(style="flex: 1; text-align:center"):
                h3("No instance Result")
                for i, data in enumerate(images_data, start=1):
                    img(src=data['semantic_pre'], style="max-width: 90%") 
                    with h3():
                        span(f"Image {i}", ":", style="color: red")
                        span(os.path.basename(data['semantic_pre']))  

    with open("gt_result.html", "w") as f:
        f.write(doc.render())


if __name__ == "__main__":
    images_data = gen_images_data(annotations_file="datasets/BUU/sample/annotations/bbox_val.json", 
                                  infer_folder="datasets/BUU/sample/val",
                                  gt_vis_save_folder="datasets/BUU/sample/val_gt/",
                                  pre_vis_folder="infer_output/BUU")
    gen_gt_result_html(images_data)
