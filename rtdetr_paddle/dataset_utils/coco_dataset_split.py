import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "rtdetr_paddle"))
import json
import numpy as np
import shutil
from datetime import datetime
from json_process import load_json_file, save_json_file
from utils.io.common import create_folder
    
    
def random_split_coco_dataset(images_folder_path, annotation_file, output_folder_path, split_info_dict):
    """
    随机划分json文件,并划分好相应的数据集
    """
    os.makedirs(output_folder_path, exist_ok=True)
    # 读取annotations.json文件
    dataset = load_json_file(annotation_file)
    # 提取images, annotations, categories
    # 随机打乱数据
    np.random.shuffle(dataset["images"])
    start_index = 0
    end_index = 0
    def filter_annotations(annotations, image_ids):
        return [ann for ann in annotations if ann["image_id"] in image_ids]
    for i, split_part in enumerate(split_info_dict.items()):
        split_part_name = split_part[0]
        ratio = split_part[1]
        if i == len(split_info_dict.items()) - 1:
            split_part_images = dataset["images"][start_index:]
        else:
            end_index += int(ratio * len(dataset["images"]))
            split_part_images = dataset["images"][start_index:end_index]
        start_index += int(ratio * len(dataset["images"]))
        split_part_folder = os.path.join(output_folder_path, split_part_name)
        os.makedirs(split_part_folder, exist_ok=True)
        for img in split_part_images:
            shutil.copy(os.path.join(images_folder_path, img["file_name"]), os.path.join(split_part_folder, img["file_name"]))
        split_part_annotations = filter_annotations(dataset["annotations"], [img["id"] for img in split_part_images])
        split_part_data = {"info": dataset["info"], "images": split_part_images, "annotations": split_part_annotations, "categories": dataset["categories"]}
        save_json_file(split_part_data, os.path.join(output_folder_path,"bbox_" + split_part_name + ".json"))
    print("数据集划分完成！")


def modify_json(json_path, assign_dataset_name):
    """
    重新划分后,修改原来json文件里面的对应图片路径
    """
    if not os.path.exists(json_path):
        raise ValueError("json path `{}` does not exists. ".format(json_path))
    with open(json_path, "r") as f:
        json_file = json.load(f)
    for img in json_file["images"]:
        img["path"] = "/datasets/" + assign_dataset_name + "/" + img["file_name"]
    with open(json_path, "w", encoding='utf-8') as f_write:
        json.dump(json_file, f_write)


def modify_label(json_path, choose_categories_name, new_dataset_name):
    if not os.path.exists(json_path):
        raise ValueError("json path `{}` does not exists. ".format(json_path))
    with open(json_path, "r") as f:
        old_json_data = json.load(f)
    # 修改images的图片的路线信息为新文件夹对应位置
    for img in old_json_data["images"]:
        img["path"] = "/datasets/" + new_dataset_name + "/" + img["file_name"]
    old_json_data["info"]["date"] = datetime.today().strftime('%Y-%m-%d')
    old_json_data["info"]["description"] = "This dataset is labeled as visible to the human eye and labeled:" + str(choose_categories_name)
    new_json_data = {
        "info" : old_json_data["info"],
        "images": old_json_data["images"],
        "annotations": [],
        "categories" : []
        }
    # 建立类别name与id连接
    categories_name_to_id = {}
    for category in old_json_data["categories"]:
        categories_name_to_id[category["name"]] = category["id"]
    # 提取出需要的name的id
    need_categories_id = []
    for category_name in choose_categories_name:
        need_categories_id.append(categories_name_to_id[category_name])
    # 新的json文件添加需要的类别信息
    for category in old_json_data["categories"]:
        if category["name"] in choose_categories_name:
            new_json_data["categories"].append(category)
    # 挑选出对应类别的annotation
    for ann in old_json_data["annotations"]:
        if ann["category_id"] in need_categories_id:
            new_json_data["annotations"].append(ann)
    with open(json_path, "w", encoding='utf-8') as f_write:
        json.dump(new_json_data, f_write)

    
def modify_datasets_annotations_label(json_file_path, choose_categories_name, new_dataset_name):
    for root, dirs, files in os.walk(json_file_path):
        for file in files:
            modify_label(os.path.join(root, file), choose_categories_name, new_dataset_name)
            print(file ,"修改标签完成")


def no_random_split_coco_dataset():
    '''
    按照别的数据集里面的json文件,来划分数据集，确保数据集的划分一样，标注的标签数量、类别可以不同
    '''
    # 数据集路径
    dataset_root = "./datasets/All_datasets/Real_Label1"
    images_folder = os.path.join(dataset_root, "images")
    annotations_path = os.path.join(dataset_root,"annotations" ,"Real_Label1.json")
    # 修改json里面的图片路径
    modify_json(annotations_path)
    # 输出路径
    output_root = os.path.join(dataset_root, "split_dataset")
    os.makedirs(output_root, exist_ok=True)
    # 读取annotations.json文件
    annotations_data = load_json_file(annotations_path)
    # 提取images, annotations, categories
    info = annotations_data["info"]
    info['date'] = datetime.today().strftime('%Y-%m-%d')
    images = annotations_data["images"]
    annotations = annotations_data["annotations"]
    categories = annotations_data["categories"]
    # 选定图片作为训练集、验证集（从指定json文件里面选）
    train_json_path = "/home/RT-DETR/rtdetr_paddle/datasets/Real_Label/annotations/bbox_train.json"
    val_json_path = "/home/RT-DETR/rtdetr_paddle/datasets/Real_Label/annotations/bbox_val.json"
    test_json_path = "/home/RT-DETR/rtdetr_paddle/datasets/Real_Label/annotations/bbox_test.json"
    def load_json(json_path):
        with open(json_path, "r") as f:
            annotations_data = json.load(f)
        info = annotations_data["info"]
        images = annotations_data["images"]
        annotations = annotations_data["annotations"]
        categories = annotations_data["categories"]
        return info, images, annotations, categories
    train_info, train_images, train_annotations, train_categories = load_json(train_json_path)
    val_info, val_images, val_annotations, val_categories = load_json(val_json_path)
    test_info, test_images, test_annotations, test_categories = load_json(test_json_path)
    # 分别为训练集、验证集和测试集创建子文件夹
    train_folder = create_folder(os.path.join(output_root, "train"))
    val_folder = create_folder(os.path.join(output_root, "val"))
    test_folder = create_folder(os.path.join(output_root, "test"))
    # 将图片文件复制到相应的子文件夹
    for img in train_images:
        shutil.copy(os.path.join(images_folder, img["file_name"]), os.path.join(train_folder, img["file_name"]))
    for img in val_images:
        shutil.copy(os.path.join(images_folder, img["file_name"]), os.path.join(val_folder, img["file_name"]))
    for img in test_images:
        shutil.copy(os.path.join(images_folder, img["file_name"]), os.path.join(test_folder, img["file_name"]))
    # 根据图片id分配annotations
    def filter_annotations(annotations, image_ids):
        return [ann for ann in annotations if ann["image_id"] in image_ids]
    train_ann = filter_annotations(annotations, [img["id"] for img in train_images])
    val_ann = filter_annotations(annotations, [img["id"] for img in val_images])
    test_ann = filter_annotations(annotations, [img["id"] for img in test_images])
    # 生成train.json, val.json, test.json
    train_json = {"info": info, "images": train_images, "annotations": train_ann, "categories": categories}
    val_json = {"info": info, "images": val_images, "annotations": val_ann, "categories": categories}
    test_json = {"info": info, "images": test_images, "annotations": test_ann, "categories": categories}
    save_json_file(train_json, os.path.join(output_root,"bbox_train.json"))
    modify_json(os.path.join(output_root,"bbox_train.json"))
    save_json_file(val_json, os.path.join(output_root,"bbox_val.json"))
    modify_json(os.path.join(output_root,"bbox_val.json"))
    save_json_file(test_json, os.path.join(output_root,"bbox_test.json"))
    modify_json(os.path.join(output_root,"bbox_test.json"))
    print("数据集划分完成！")


def assign_images_split(input_root, annotation_file, output_root):
    """
    手动划分数据集后,根据图片名称划分json文件
    """
    # 输出路径
    os.makedirs(output_root, exist_ok=True)
    # 读取annotations.json文件
    annotations_data = load_json_file(annotation_file)
    # 提取images, annotations, categories
    images = annotations_data["images"]
    annotations = annotations_data["annotations"]
    categories = annotations_data["categories"]

    def get_images(assign_images_path, images):
        assign_images = []
        for root, dirs, files in os.walk(assign_images_path):
            for img in images:
                if img["file_name"] in files:
                    assign_images.append(img)
        return assign_images
    
    # 根据图片id分配annotations
    def filter_annotations(annotations, image_ids):
        return [ann for ann in annotations if ann["image_id"] in image_ids]
    
    assign_train_images_path = os.path.join(input_root, "train")
    train_images = get_images(assign_train_images_path, images)
    train_ann = filter_annotations(annotations, [img["id"] for img in train_images])
    train_json = {"info": annotations_data["info"], "images": train_images, "annotations": train_ann, "categories": categories}
    save_json_file(train_json, os.path.join(output_root,"bbox_train.json"))

    assign_val_images_path = os.path.join(input_root, "val")
    val_images = get_images(assign_val_images_path, images)
    val_ann = filter_annotations(annotations, [img["id"] for img in val_images])
    val_json = {"info": annotations_data["info"], "images": val_images, "annotations": val_ann, "categories": categories}
    save_json_file(val_json, os.path.join(output_root,"bbox_val.json"))

    if os.path.exists(os.path.join(input_root, "test")):
        assign_test_images_path = os.path.join(input_root, "test")
        test_images = get_images(assign_test_images_path, images)
        test_ann = filter_annotations(annotations, [img["id"] for img in test_images])
        test_json = {"info": annotations_data["info"], "images": test_images, "annotations": test_ann, "categories": categories}
        save_json_file(test_json, os.path.join(output_root,"bbox_test.json"))

    print("数据集划分完成！")


def check_split_complete(dataset_path1, dataset_path2):
    train_val_test = []
    for root1, dirs1, files1 in os.walk(dataset_path1):
        for file1 in files1:
            train_val_test.append(file1)
    sorted_files1 = sorted(train_val_test)
    print(sorted_files1)
    print(len(sorted_files1))
    files2 = os.listdir(dataset_path2)
    sorted_files2 = sorted(files2)
    print(sorted_files2)
    print(len(sorted_files2))
    # 判断是否相同
    if sorted_files1 == sorted_files2:
        print("complete same")
    # 不相同找出不相同的文件
    else:
        for file in sorted_files1:
            if file not in sorted_files2:
                print(file, "can not find in", dataset_path2)
        for file in sorted_files2:
            if file not in sorted_files1:
                print(file, "can not find in", dataset_path1)


def split_train_dataset_to_no_label_and_label(train_folder, train_no_label_folder, train_label_folder):
    for file in os.listdir(train_folder):
        if file not in os.listdir(train_label_folder):
            shutil.copy(os.path.join(train_folder, file), os.path.join(train_no_label_folder, file))


def TD20240705_LA_dataset_split():
    """
    手动划分好图片再分数据集json
    """
    fracture = {"bimeihua": 4, "caiyumei": 6, "dingjunmei": 4, "lijiling": 14, "liusuzhen": 14, 
                "miaoqilan": 18, "wangxiuhua": 18, "wanjiarong": 18, "zhangyebao": 16, "zhoukelan": 18}
    normal = {"hukaixiu": 15, "lishuying": 16, "liujiacai": 6, "liyan": 14, "peizongping": 18, 
              "pengcuilian": 10, "weilanhua": 15, "qiaojigang": 18, "yangdongmei": 11, "zhangcuiyyy":3}
    
    train = {"fracture": ["bimeihua", "lijiling", "wangxiuhua", "caiyumei", "zhangyebao", "dingjunmei"],
             "normal": ["hukaixiu", "liujiacai", "pengcuilian", "weilanhua", "yangdongmei", "zhangcuiying", "zhangcuiyyy"]}
    
    val = {"fracture": ["wanjiarong", "liusuzhen"], 
           "normal": ["lishuying", "peizongping"]}


def KFoldsplit_dataset_according_case(fracture_folder, normal_folder, annotation_file, output_folder):
    """
    根据病人划分数据集
    """
    fracture = {"bimeihua": 4, "caiyumei": 6, "dingjunmei": 4, "lijiling": 14, "liusuzhen": 14, 
                "miaoqilan": 18, "wangxiuhua": 18, "wanjiarong": 18, "zhangyebao": 16, "zhoukelan": 18}
    normal = {"hukaixiu": 15, "lishuying": 16, "liujiacai": 6, "liyan": 14, "peizongping": 18, 
              "pengcuilian": 10, "weilanhua": 15, "qiaojigang": 18, "yangdongmei": 11, "zhangcuiyyy":3}
    
    fold1_val_case_list = ["bimeihua", "miaoqilan", "hukaixiu", "pengcuilian"]

    fold2_val_case_list = ["caiyumei", "wangxiuhua", "lishuying", "weilanhua"]

    fold3_val_case_list = ["dingjunmei", "wanjiarong", "liujiacai", "qiaojigang"]

    fold4_val_case_list = ["lijiling", "zhangyebao", "liyan", "yangdongmei"]

    fold5_val_case_list = ["liusuzhen", "zhoukelan", "peizongping", "zhangcuiyyy"]

    fold1_val_folder = create_folder(os.path.join(output_folder, "fold1", "val"))
    fold1_train_folder = create_folder(os.path.join(output_folder, "fold1", "train"))
    fold2_val_folder = create_folder(os.path.join(output_folder, "fold2", "val"))
    fold2_train_folder = create_folder(os.path.join(output_folder, "fold2", "train"))
    fold3_val_folder = create_folder(os.path.join(output_folder, "fold3", "val"))
    fold3_train_folder = create_folder(os.path.join(output_folder, "fold3", "train"))
    fold4_val_folder = create_folder(os.path.join(output_folder, "fold4", "val"))
    fold4_train_folder = create_folder(os.path.join(output_folder, "fold4", "train"))
    fold5_val_folder = create_folder(os.path.join(output_folder, "fold5", "val"))
    fold5_train_folder = create_folder(os.path.join(output_folder, "fold5", "train"))

    def copy_images(fracture_folder, normal_folder, fold_val_case_list, output_val_folder, output_train_folder):
        for filename in os.listdir(fracture_folder):
            filename_with_no_ext = filename.split("_")[0]
            if filename_with_no_ext in fold_val_case_list:
                shutil.copy(os.path.join(fracture_folder, filename), os.path.join(output_val_folder,filename))
            else:
                shutil.copy(os.path.join(fracture_folder, filename), os.path.join(output_train_folder,filename))

        for filename in os.listdir(normal_folder):
            filename_with_no_ext = filename.split("_")[0]
            if filename_with_no_ext in fold_val_case_list:
                shutil.copy(os.path.join(normal_folder, filename), os.path.join(output_val_folder,filename))
            else:
                shutil.copy(os.path.join(normal_folder, filename), os.path.join(output_train_folder,filename))

    copy_images(fracture_folder, normal_folder, fold1_val_case_list, fold1_val_folder, fold1_train_folder)
    copy_images(fracture_folder, normal_folder, fold2_val_case_list, fold2_val_folder, fold2_train_folder)
    copy_images(fracture_folder, normal_folder, fold3_val_case_list, fold3_val_folder, fold3_train_folder)
    copy_images(fracture_folder, normal_folder, fold4_val_case_list, fold4_val_folder, fold4_train_folder)
    copy_images(fracture_folder, normal_folder, fold5_val_case_list, fold5_val_folder, fold5_train_folder)

    assign_images_split(os.path.join(output_folder, "fold1"), annotation_file, os.path.join(output_folder, "fold1", "annotations"))
    assign_images_split(os.path.join(output_folder, "fold2"), annotation_file, os.path.join(output_folder, "fold2", "annotations"))
    assign_images_split(os.path.join(output_folder, "fold3"), annotation_file, os.path.join(output_folder, "fold3", "annotations"))
    assign_images_split(os.path.join(output_folder, "fold4"), annotation_file, os.path.join(output_folder, "fold4", "annotations"))
    assign_images_split(os.path.join(output_folder, "fold5"), annotation_file, os.path.join(output_folder, "fold5", "annotations"))


if __name__ == "__main__":
    # no_random_split_coco_dataset()
    # split_train_dataset_to_no_label_and_label("datasets/miccai/xray/images/train_instance",
    #                                           "datasets/miccai/xray/images/train_instance_60",
    #                                           "datasets/miccai/xray/images/train_instance_20")
    # random_split_coco_dataset("datasets/LA_xray_fracture/images",
    #                           "datasets/LA_xray_fracture/annotations/LA_xray_fracture.json",
    #                           "datasets/LA_xray_fracture/split_dataset",
    #                           {"train": 0.7, "val": 0.3})

    # assign_images_split("datasets/TD20240705_LA/split_dataset", "TD20240705_LA_coco.json","datasets/TD20240705_LA/split_dataset")
    KFoldsplit_dataset_according_case("datasets/TD20240705_LA/fracture", 
                                      "datasets/TD20240705_LA/normal", 
                                      "datasets/TD20240705_LA/split_dataset/annotations/TD20240705_LA_coco.json", 
                                      "datasets/TD20240705_LA/split_dataset")