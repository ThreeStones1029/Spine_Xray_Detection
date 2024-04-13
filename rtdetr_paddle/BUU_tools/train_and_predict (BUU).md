<!--
 * @Descripttion: 
 * @version: 
 * @Author: ShuaiLei
 * @Date: 2023-10-24 10:22:24
 * @LastEditors: ShuaiLei
 * @LastEditTime: 2024-03-12 10:22:18
-->
# 前期准备
## 0.1.数据标注
### windows
### ubuntu系统


# 一、模型训练、评估与预测
## 1.1.训练命令
~~~bash
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml --eval --use_vdl=true --vdl_log_dir=vdl_dir/BUU/instance
~~~
## 1.2.训练可视化命令
~~~bash
visualdl --logdir vdl_dir/BUU/instance --host 0.0.0.0
~~~

## 1.3.测试命令
~~~bash
python tools/infer.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
~~~

## 1.4.评估命令
~~~bash
python tools/eval.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -o weights=output/BUU_Model/instance/rtdetr_r50vd_6x_coco/best_model.pdparams
~~~
