'''
Description: 
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-13 15:17:15
LastEditors: ShuaiLei
LastEditTime: 2024-04-13 15:18:06
'''
import os


def create_folder(path):
    os.makedirs(path, exist_ok=True)
    return path

def join(*args):
    return os.path.join(*args)