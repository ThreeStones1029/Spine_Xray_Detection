'''
Description: 用于从公司发过来的数据中挑选骨折以及完全正常的X线片
version: 1.0
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-07-11 07:25:36
LastEditors: ShuaiLei
LastEditTime: 2024-07-11 15:21:10
'''
import os
import shutil


def xray_choose(dataset_folder, choosed_id_list, fracture_name_list, normal_name_list, save_folder):
    """
    先挑选需要的下标，然后手动删除不清楚的
    """
    noraml_folder = os.path.join(save_folder, "normal")
    fracture_folder = os.path.join(save_folder, "fracture")
    for subfolder_name in fracture_name_list:
        for id in choosed_id_list:
            new_filename = subfolder_name.split(" ")[0] + "_" + str(id) + ".tif"
            shutil.copy(os.path.join(dataset_folder, subfolder_name, str(id) + ".tif"), os.path.join(fracture_folder, new_filename))


    for subfolder_name in normal_name_list:
        for id in choosed_id_list:
            new_filename = subfolder_name.split(" ")[0] + "_" + str(id) + ".tif"
            shutil.copy(os.path.join(dataset_folder, subfolder_name, str(id) + ".tif"), os.path.join(noraml_folder, new_filename))


if __name__ == "__main__":
    xray_choose(dataset_folder="/root/share/ShuaiLei/RT-DETR/rtdetr_paddle/datasets/trainData",
                choosed_id_list=[1, 5, 10, 15, 20, 25, 30, 35, 40, 380, 385, 390, 395, 400, 405, 410, 415, 420],
                fracture_name_list=["bimeihua", "caiyumei XO", "dingjunmei",  "jiangfang", "lijiling", "liusuzhen", "liyuefang", "miaoqilan", "wangxiuhua",
                                    "wanjiarong", "zhoukelan X"],
                normal_name_list=["dukemei", "hukaixiu XO", "lishuying", "liyan", "liujiacai", "peizongping", "pengcuilian", "qiaojigang X", "weilanhua", "yangdongmei", 
                                  "zhangcuiying", "zhangcuiyy X", "zhangcuiyyy", "zhangyebao X"],
                save_folder="datasets/TD_LA_fracture")