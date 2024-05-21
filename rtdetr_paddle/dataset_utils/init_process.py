'''
Descripttion: 
version: 
Author: ShuaiLei
Date: 2023-11-07 17:17:44
LastEditors: ShuaiLei
LastEditTime: 2023-11-10 15:05:17
'''
import os
from PIL import Image
import numpy as np


class InitProcess:
    def __init__(self, images_folder, save_images_folder):
        """params"""
        self.images_folder = images_folder
        self.save_images_folder = save_images_folder
        self.crop_flip_images()


    def crop_flip_images(self):
        for root, dirs, files in os.walk(self.images_folder):
            for file in files:
                image_path = os.path.join(root,file)
                self.crop_flip_image(image_path)


    def crop_flip_image(self, image_path):
        image = Image.open(image_path)
        image_array = np.array(image)
        file_name = image_path.split('/')[-2] + '_' + os.path.basename(image_path)
        black_pixels = np.argwhere(np.all(image_array == [0, 0, 0], axis=-1))
        if black_pixels.size > 0:
            top, left = black_pixels.min(axis=0)
            bottom, right = black_pixels.max(axis=0)
            cropped_image = image.crop((left, top, right, bottom))
            fliped_cropped_image = self.conver_image(cropped_image)
            fliped_cropped_image.save(os.path.join(self.save_images_folder, file_name))


    def conver_image(self, image):
        image_array = np.array(image)
        max_pixel_value = image_array.max()
        new_imaeg_array = max_pixel_value - image_array
        new_image = Image.fromarray(new_imaeg_array)
        return new_image


def conver_images(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            image_path = os.path.join(root, file)
            image = Image.open(image_path)
            image_array = np.array(image)
            max_pixel_value = image_array.max()
            print(max_pixel_value)
            new_imaeg_array = max_pixel_value - image_array
            new_image = Image.fromarray(new_imaeg_array)
            new_image.save(os.path.join(output_folder, file))


if __name__ == "__main__":
    # InitProcess("datasets/x_ray2023_11_05", "datasets/x_ray2023_11_05")
    conver_images("datasets/XJT", "datasets/XJT")