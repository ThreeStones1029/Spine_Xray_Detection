'''
Description: 
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-16 02:22:37
LastEditors: ShuaiLei
LastEditTime: 2024-04-16 07:30:07
'''
import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()
    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'
    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.val()
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default="configs/rtdetr/rtdetr_r50vd_6x_coco.yml")
    parser.add_argument('--resume', '-r', type=str, default="output/test/rtdetr_r50vd_6x_coco/best_checkpoint.pth")
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--amp', action='store_true', default=False,)
    args = parser.parse_args()
    main(args)