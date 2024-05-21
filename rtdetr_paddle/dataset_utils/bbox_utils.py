'''
Descripttion: 
version: 
Author: ThreeStones1029 221620010039@hhu.edu.cn
Date: 2023-09-23 15:57:51
LastEditors: ShuaiLei
LastEditTime: 2024-03-01 10:38:07
'''
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor
import matplotlib.pyplot as plt
from copy import copy
import math


# 判断顶点是否在框内
def point_in_bbox(point_x, point_y, ann):
    xmin = ann["bbox"][0]
    xmax = ann["bbox"][0] + ann["bbox"][2]
    ymin = ann["bbox"][1]
    ymax = ann["bbox"][1] + ann["bbox"][3]
    if point_x > xmin and point_x < xmax:
        if point_y > ymin and point_y < ymax:
            return True
    return False


# 求bbox中心
def get_bbox_center(ann):
    x = ann["bbox"][0] + ann["bbox"][2] / 2
    y = ann["bbox"][1] + ann["bbox"][3] / 2
    return x, y 


# 判断两个bbox是否相交
def judgment_bboxes_intersection(ann1, ann2):
    w1, h1 = ann1['bbox'][2], ann1['bbox'][3]
    w2, h2 = ann2['bbox'][2], ann2['bbox'][3]
    bbox1_center_x, bbox1_center_y = get_bbox_center(ann1)
    bbox2_center_x, bbox2_center_y = get_bbox_center(ann2)
    bbox_center_distance_x = abs(bbox1_center_x - bbox2_center_x)
    bbox_center_distance_y = abs(bbox1_center_y - bbox2_center_y)
    if bbox_center_distance_x < (w1 + w2)/2.0 and bbox_center_distance_y < (h1 + h2)/2.0:
        return True
    else:
        return False
    

def bbox_iou(ann1, ann2, eps=1e-9):
    """calculate the iou of box1 and box2"""
    x1, y1, x2, y2 = ann1['bbox'][0], ann1['bbox'][1], ann1['bbox'][0] + ann1['bbox'][2], ann1['bbox'][1] + ann1['bbox'][3]
    x3, y3, x4, y4 = ann2['bbox'][0], ann2['bbox'][1], ann2['bbox'][0] + ann2['bbox'][2], ann2['bbox'][1] + ann2['bbox'][3]
    # 先判断两个框是否相交 
    is_bboxes_intersection = judgment_bboxes_intersection(ann1, ann2)
    if is_bboxes_intersection:
        x_inter1 = max(x1, x3)
        y_inter1 = max(y1, y3)
        x_inter2 = min(x2, x4)
        y_inter2 = min(y2, y4)
        overlap = abs(x_inter2-x_inter1) * abs(y_inter2 - y_inter1)
        area1 = ann1['area'] if 'area' in ann1.keys() else abs(ann1['bbox'][2] * ann1['bbox'][3])
        area2 = ann2['area'] if 'area' in ann2.keys() else abs(ann2['bbox'][2] * ann2['bbox'][3])
        union = area1 + area2 - overlap + eps
        iou1 = overlap / area1
        iou2 = overlap / area2
        return iou1 if iou1 > iou2 else iou2
    else:
        return 0


# 计算点到线的距离
def point2line_distance(k, b, point):
    x = point[0]
    y = point[1]
    distance = abs((x * k + b - y) / (k ** 2 + 1)**0.5)
    return distance

# 计算点到框中心的距离
def get_point2bbox_center_distance(ann, point_x, point_y):
    center_x, center_y = get_bbox_center(ann)
    distance = math.sqrt((center_x - point_x) * (center_x - point_x) + (center_y - point_y) * (center_y - point_y))
    return distance


# 框中心到线的距离字典
def get_bboxes_center_distance2_line(anns, k, b):
    distance_to_anns = {}
    for ann in anns:
        bbox_center_x, bbox_center_y = get_bbox_center(ann)
        distance = point2line_distance(k, b, [bbox_center_x, bbox_center_y])
        distance_to_anns[distance] = ann
    return distance_to_anns


# 拟合直线
def fitted_line(anns):
    X = []
    Y = []
    for ann in anns:
        X.append(ann["bbox"][0] + ann["bbox"][2] / 2)
        Y.append(ann["bbox"][1] + ann["bbox"][3] / 2)
    # 输入数据
    X_train = np.array(X).reshape((len(X), 1))
    Y_train = np.array(Y).reshape((len(Y), 1))

    # # 线性回归拟合一条y = k * x + b的直线
    # lineModel_1 = LinearRegression()
    # lineModel_1.fit(X_train, Y_train)

    # Y_predict = lineModel_1.predict(X_train)

    # # 获取斜率和截距
    # k1 = lineModel_1.coef_[0][0]
    # b1 = lineModel_1.intercept_[0]

    # RANSAC拟合一条直线
    ransac = RANSACRegressor()
    ransac.fit(X_train, Y_train)
    k2 = ransac.estimator_.coef_[0][0]
    b2 = ransac.estimator_.intercept_[0]
    return k2, b2


# 计算直线的两个端点坐标
def get_line_point(k, b,min_y, max_y):
    x1 = (min_y - b) / k
    x2 = (max_y - b) / k
    return (x1, min_y), (x2, max_y)


# 得到垂足坐标
def get_pedal_point(k, b, m, n):
    x = (m + n * k- b * k)/ (k ** 2 + 1)
    y = x * k + b
    return x, y


# 判断两点是否在直线同一侧 (A * x1 + B * y1 + C)* (A * x2 + B * y2 + C)> 0
def judgment_points_on_same_side_line(points, k, b):
    point1_x, point1_y = points[0][0], points[0][1]
    point2_x, point2_y = points[1][0], points[1][1]
    if (k * point1_x - point1_y + b) * (k * point2_x - point2_y + b) > 0:
        return True
    else:
        return False
    
# 返回点离一系列框中最近的框
def get_nearest_ann(anns, point_x, point_y):
    min_distance = get_point2bbox_center_distance(anns[0], point_x, point_y)
    nearest_ann = anns[0]
    for ann in anns:
        point2bbox_center_distance = get_point2bbox_center_distance(ann, point_x, point_y)
        if point2bbox_center_distance < min_distance:
            min_distance = point2bbox_center_distance
            nearest_ann = ann
    return nearest_ann


def judgment_point_on_line_side(point, k, b):
    point_x, point_y = point[0], point[1]
    if (k * point_x - point_y + b) > 0:
        return True
    else:
        return False
    

# 计算两条直线的交点
def get_lines_intersection_point(k1, b1, k2, b2):
    point_x = (b2 - b1) / (k1 - k2)
    point_y = (k1 * b2 - k2 * b1) / (k1 - k2)
    return point_x, point_y


# 计算两个bbox高的一半和
def compute_two_anns_h_sum(ann1, ann2):
    bbox1 = ann1["bbox"]
    bbox2 = ann2["bbox"]
    h_sum = bbox1[3] / 2 + bbox2[3] / 2
    return h_sum


# 计算两个bbox中心之间距离
def compute_two_anns_center_distance(ann1, ann2):
    bbox1_center_x, bbox1_center_y = get_bbox_center(ann1)
    bbox2_center_x, bbox2_center_y = get_bbox_center(ann2)
    return bbox2_center_y - bbox1_center_y


# 对多个bbox从上到下排序
def sort_anns(anns):
    sorted_id2ann = {}
    bboxes_center_y2ann = {}
    bboxes_h = []
    for ann in anns:
        bbox_center_x, bbox_center_y = get_bbox_center(ann)
        bboxes_center_y2ann[bbox_center_y] = ann
        bboxes_h.append(ann["bbox"][3])
    for i, bbox_center_y in enumerate(sorted(bboxes_center_y2ann.keys())):
        sorted_id2ann[i + 1] = bboxes_center_y2ann[bbox_center_y]
    mean_bbox_h = np.mean(bboxes_h)
    return sorted_id2ann, mean_bbox_h


def compute_the_coordinates_of_the_bisector_between_two_points(point1, point2, k):
    """
    k等分点靠近point1,例如x1, y1, x2, y2靠近x1,y1的三分之一等分点
    x = 2/3 * x1 + 1/3 * x2
    y = 2/3 * y1 + 1/3 * y2
    """
    x1, y1 = point1
    x2, y2 = point2
    x = (1 - k) * x1 + k * x2
    y = (1 - k) * y1 + k * y2
    return x, y


def add_anns(ann1, ann2, add_number):
    anns = []
    bbox1 = ann1["bbox"]
    bbox2 = ann2["bbox"]
    for i in range(add_number):
        ann = copy(ann1)
        point1 = (bbox1[0], bbox1[1])
        point2 = (bbox2[0], bbox2[1])
        point3 = ((bbox1[0] + bbox1[2]), (bbox1[1] + bbox1[3]))
        point4 = ((bbox2[0] + bbox2[2]), (bbox2[1] + bbox2[3]))
        min_x, min_y = compute_the_coordinates_of_the_bisector_between_two_points(point1, point2, (i + 1) / (add_number + 1))
        max_x, max_y = compute_the_coordinates_of_the_bisector_between_two_points(point3, point4, (i + 1) / (add_number + 1))
        w = max_x - min_x
        h = max_y - min_y
        bbox = [min_x, min_y, w, h]
        ann["bbox"] = bbox
        ann["score"] = 0.9
        anns.append(ann)
    return anns


# 计算排好序的相邻bbox距离
def add_miss_vertebrae_anns(sorted_id2ann, mean_bbox_h):
    final_postprocess_single_img_vertebrae_results = []
    number_anns = len(sorted_id2ann.keys())
    for i in range(1, number_anns):
        ann1 = sorted_id2ann[i]
        ann2 = sorted_id2ann[i + 1]
        two_adjacent_bboxes_h_sum = compute_two_anns_h_sum(ann1, ann2)
        two_adjacent_bboxes_center_distance = compute_two_anns_center_distance(ann1, ann2)
        print("two_adjacent_bboxes_center_distance: ",two_adjacent_bboxes_center_distance )
        print("two_adjacent_bboxes_h_sum: ", two_adjacent_bboxes_h_sum)
        # 计算两个相邻bbox的中心点y距离以及框的高度和的一半
        min_distance_between_adjacent_bboxes = two_adjacent_bboxes_center_distance - two_adjacent_bboxes_h_sum
        # 没有间距
        if min_distance_between_adjacent_bboxes <= 0:
            final_postprocess_single_img_vertebrae_results.append(sorted_id2ann[i]) 
        # 有间距,但间距很小
        if min_distance_between_adjacent_bboxes > 0 and min_distance_between_adjacent_bboxes < mean_bbox_h / 3:
            final_postprocess_single_img_vertebrae_results.append(sorted_id2ann[i]) 
        # 有间距,间距差不多可以补充一个ann
        if min_distance_between_adjacent_bboxes > mean_bbox_h / 3 and min_distance_between_adjacent_bboxes < 9 / 8 * mean_bbox_h:
            will_be_add_anns = add_anns(ann1, ann2, 1)
            final_postprocess_single_img_vertebrae_results.append(sorted_id2ann[i])
            final_postprocess_single_img_vertebrae_results.append(will_be_add_anns[0])
        # 有间距,补充多个ann
        if min_distance_between_adjacent_bboxes >= 9/8 * mean_bbox_h:
            add_anns_number = np.clip(min_distance_between_adjacent_bboxes / mean_bbox_h)
            will_be_add_anns = add_anns(ann1, ann2, add_anns_number)
            final_postprocess_single_img_vertebrae_results.append(sorted_id2ann[i])
            for j in range(add_anns_number):
                final_postprocess_single_img_vertebrae_results.append(will_be_add_anns[j])
    # 添加最后一个
    final_postprocess_single_img_vertebrae_results.append(sorted_id2ann[number_anns])
    return final_postprocess_single_img_vertebrae_results
            

if __name__ == '__main__':
    # bbox1 = {"bbox":[332.0986022949219, 645.1433715820312, 212.19015502929688, 183.3194580078125]}
    # bbox2 = {"bbox":[334.4254150390625, 644.0218505859375, 209.51202392578125, 186.89093017578125]}
    # iou = bbox_iou(bbox1, bbox2)
    # print(iou)
    anns163 = [{'image_id': 163, 'category_id': 1, 'bbox': [689.5127563476562, 10.819843292236328, 260.33984375, 205.33568954467773], 'score': 0.8296758532524109, 'file_name': 'bimeihua60.png', 'category_name': 'vertebrae'},
            {'image_id': 163, 'category_id': 1, 'bbox': [592.2443237304688, 145.32034301757812, 249.73297119140625, 222.6868896484375], 'score': 0.8886268138885498, 'file_name': 'bimeihua60.png', 'category_name': 'vertebrae'},
            {'image_id': 163, 'category_id': 1, 'bbox': [455.7540283203125, 526.3576049804688, 244.69775390625, 191.6048583984375], 'score': 0.917931318283081, 'file_name': 'bimeihua60.png', 'category_name': 'vertebrae'},
            {'image_id': 163, 'category_id': 1, 'bbox': [452.194091796875, 705.279541015625, 218.16632080078125, 166.27874755859375], 'score': 0.9243277311325073, 'file_name': 'bimeihua60.png', 'category_name': 'vertebrae'}]
    anns165 = [{'image_id': 165, 'category_id': 1, 'bbox': [373.45623779296875, 156.12428283691406, 284.7904052734375, 187.92640686035156], 'score': 0.5359383225440979, 'file_name': 'dingjunmei180.png', 'category_name': 'vertebrae'}, 
            {'image_id': 165, 'category_id': 1, 'bbox': [315.0342102050781, 453.73614501953125, 290.3872375488281, 206.60125732421875], 'score': 0.682439386844635, 'file_name': 'dingjunmei180.png', 'category_name': 'vertebrae'},
            {'image_id': 165, 'category_id': 1, 'bbox': [391.1253967285156, 3.7736542224884033, 320.9715881347656, 163.02181696891785], 'score': 0.8103837370872498, 'file_name': 'dingjunmei180.png', 'category_name': 'vertebrae'}, 
            {'image_id': 165, 'category_id': 1, 'bbox': [297.8074951171875, 596.95166015625, 303.7999267578125, 175.21746826171875], 'score': 0.847192108631134, 'file_name': 'dingjunmei180.png', 'category_name': 'vertebrae'},
            {'image_id': 165, 'category_id': 1, 'bbox': [238.9594268798828, 756.6455688476562, 341.5387420654297, 211.09881591796875], 'score': 0.88099604845047, 'file_name': 'dingjunmei180.png', 'category_name': 'vertebrae'}]
    anns174 = [{'image_id': 174, 'category_id': 1, 'bbox': [326.7227783203125, 822.7379150390625, 282.37762451171875, 151.7772216796875], 'score': 0.8888289928436279, 'file_name': 'peizongping180.png', 'category_name': 'vertebrae'}, 
               {'image_id': 174, 'category_id': 1, 'bbox': [372.1904296875, -0.42956870794296265, 196.93804931640625, 98.85431236028671], 'score': 0.8978489637374878, 'file_name': 'peizongping180.png', 'category_name': 'vertebrae'}, 
               {'image_id': 174, 'category_id': 1, 'bbox': [367.6080627441406, 107.38914489746094, 223.43179321289062, 147.65167236328125], 'score': 0.9377333521842957, 'file_name': 'peizongping180.png', 'category_name': 'vertebrae'}, 
               {'image_id': 174, 'category_id': 1, 'bbox': [348.40191650390625, 441.84429931640625, 253.6907958984375, 166.90576171875], 'score': 0.9393659830093384, 'file_name': 'peizongping180.png', 'category_name': 'vertebrae'}, 
               {'image_id': 174, 'category_id': 1, 'bbox': [334.8031005859375, 623.2579956054688, 259.58306884765625, 179.50665283203125], 'score': 0.941200315952301, 'file_name': 'peizongping180.png', 'category_name': 'vertebrae'}, 
               {'image_id': 174, 'category_id': 1, 'bbox': [356.7311096191406, 265.51373291015625, 246.19100952148438, 165.14035034179688], 'score': 0.9413239359855652, 'file_name': 'peizongping180.png', 'category_name': 'vertebrae'}]
    sorted_id2ann, mean_bbox_h = sort_anns(anns174)   
    final_postprocess_single_img_vertebrae_results = add_miss_vertebrae_anns(sorted_id2ann, mean_bbox_h)
    print(final_postprocess_single_img_vertebrae_results)