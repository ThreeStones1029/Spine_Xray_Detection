'''
Description: 
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-16 02:22:37
LastEditors: ShuaiLei
LastEditTime: 2024-04-17 08:37:24
'''
import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import json
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def prepare_coco_json(test_dataset, train_dataset):
    """
    test_dataloader will be used to generate json which contain imageid and filename used to infer.
    """
    if len(os.listdir(test_dataset["img_folder"])) > 0:
        test_data = {"images": []}
        for i, file_name in enumerate(os.listdir(test_dataset["img_folder"])):
            test_data["images"].append({"id": i,
                                        "file_name": file_name})
        # print(os.path.abspath(train_dataset["ann_file"]))
        # print(train_dataset["ann_file"])
        with open(os.path.abspath(train_dataset["ann_file"]), "r") as f:
            train_data = json.load(f)
        test_data["categories"] = train_data["categories"]

        with open("rtdetr_pytorch/infer.json", "w") as f:
            json.dump(test_data, f)


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()
    assert not all([args.tuning, args.resume]), 'Only support from_scrach or resume or tuning at one time'    
    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )
    if "test_dataloader" in cfg.yaml_cfg.keys() and "train_dataloader" in cfg.yaml_cfg.keys():
        prepare_coco_json(cfg.yaml_cfg["test_dataloader"]["dataset"], cfg.yaml_cfg["train_dataloader"]["dataset"])

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.infer(args.infer_output_dir)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default="rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml")
    parser.add_argument('--resume', '-r', type=str, default="rtdetr_pytorch/output/test/rtdetr_r50vd_6x_coco/best_checkpoint.pth")
    parser.add_argument('--infer_output_dir', type=str, default="rtdetr_pytorch/infer_output/test")
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--amp', action='store_true', default=False,)
    args = parser.parse_args()
    main(args)