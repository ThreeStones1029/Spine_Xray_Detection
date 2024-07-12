'''
Description: 
version: 
Author: ThreeStones1029 221620010039@hhu.edu.cn
Date: 2024-01-17 17:10:49
LastEditors: ShuaiLei
LastEditTime: 2024-07-12 01:40:47
'''
import cv2
import os
import numpy as np
from PIL import Image
import nibabel as nib
from tqdm import tqdm


def conver_tiff2png(input_path, output_path):
    """
    The function will be used to conver tiff format to png format.
    param: input_path: The image with tiff format.
    param: output_path: The image with png format.
    """
    tiff_image = cv2.imread(input_path)
    cv2.imwrite(output_path, tiff_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def conver_tiffs2pngs(input_folder, output_folder):
    """
    The function will be used to conver the images with tiff format in input folder to images with pngs format.
    param: input_folder: The tiff images folder.
    param: output_folder: The png images folder.
    """
    for file in tqdm(os.listdir(input_folder), desc="convering"):
        filename_no_ext = file.split(".")[0]
        png_filename = filename_no_ext + ".png"
        conver_tiff2png(os.path.join(input_folder, file), os.path.join(output_folder, png_filename))


def blackwhite_inverse_images(input_folder, output_folder):
    """
    The function will flip black and white.
    param: input_folder: Black vertebrae on a white background images.
    param: output_folder: white vertebrae on a black background images.
    """
    for root, dirs, files in tqdm(os.walk(input_folder), desc="inversing"):
        for file in files:
            if file.endswith(".png") or file.endswith(".bmp") or file.endswith(".jpg") or file.endswith(".jpeg"):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_folder, file)
                blackwhite_inverse_image(input_path, output_path)


def blackwhite_inverse_image(input_path, output_path):
    """
    The function will flip black and white.
    param: input_path: Black vertebrae on a white background image.
    param: output_path: white vertebrae on a black background image.
    """
    image = Image.open(input_path)
    image_array = np.array(image)
    max_pixel_value = image_array.max()
    new_imaeg_array = max_pixel_value - image_array
    new_image = Image.fromarray(new_imaeg_array)
    new_image.save(output_path)


def leftright_flip(image):
    """
    The function will be used to flip image from left to right.
    param: image: the image will be used to flip.
    """
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return flipped_image


def top_bottom_flip(image):
    """
    The function will be used to flip image from bottom to top.
    param: image: The image will be used to flip .
    """
    flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return flipped_image


def rotate_images(input_folder, output_folder):
    """
    The function will be used to rotate images up and down.
    param: input_folder: The origin images folder.
    param: output_folder: The rotated imaegs folder.
    """
    for root, dirs, files in tqdm(os.walk(input_folder), desc="rotating"):
        for file in files:
            image_path = os.path.join(root, file)
            image = Image.open(image_path)
            rotated_image = image.rotate(180)
            rotated_image.save(os.path.join(output_folder, file))


def rotate_image(image, angle):
    """
    The function will be used to rotate image
    param: image: The image will be rotated.
    param: angle: The rotated angle.
    """
    rotated_image = image.rotate(angle)
    return rotated_image


def adaptive_histogram_equalization(image, block_size=32):
    """
    The function will be used to preprocess image with histogram.
    param: image: The image will be used to process.
    param: block_size: the histogram parameter.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(block_size, block_size))
    equalized = clahe.apply(gray)
    result = cv2.merge([equalized] * 3)
    return result


def crop_image_black_edge(input_path, output_path):
    """
    The function will be used to crop image black edge.
    param: input_path: the image will be used to crop black edge.
    param: output_path: the cropped image save path.
    """
    image = Image.open(input_path)
    bbox = image.getbbox()
    cropped_image = image.crop(bbox)
    cropped_image.save(output_path)


def conver_nii2png(input_path, output_path):
    """
    The function will be used to conver nii.gz file to png.
    param: input_path: The nii image path.
    param: output_path: The png image save path.
    """
    nii_image = nib.load(input_path)
    nii_data = nii_image.get_fdata()
    image_array = np.squeeze(nii_data)
    image_array = np.uint8((image_array - image_array.min()) / ((image_array.max() - image_array.min())) * 255)
    image = Image.fromarray(image_array)
    image = image.convert("L")
    flipped_image = top_bottom_flip(image)
    flipped_image.save(output_path)


def conver_niis2pngs(input_folder, output_folder):
    """
    The function will be used conver nii files to pngs
    param: input_folder: The nii images folder.
    param: output_folder: The png images folder.
    """
    for file in os.listdir(input_folder):
        if file.endswith("nii.gz"):
            filename_no_ext = file.split(".")[0]
            png_filename = filename_no_ext + ".png"
            conver_nii2png(os.path.join(input_folder, file), os.path.join(output_folder, png_filename))

    
if __name__ == "__main__":
    # image_path = "ABLSpineLevelCheck/datasets/TD2/peizongping180.png"
    # original_image = cv2.imread(image_path)
    # # 自适应直方图均衡化
    # result_image = adaptive_histogram_equalization(original_image)
    # # 应用高斯滤波
    # result_image = cv2.GaussianBlur(result_image, (5, 5), 0)
    # cv2.imwrite("/home/ABLSpineLevelCheck/datasets/TD2/result.png", result_image)

    ## vertDetect2d-62_20240226
    # conver_tiffs2pngs("datasets/vertDetect2d-62_20240226", "datasets/vertDetect2d-62_20240226")
    # blackwhite_inverse_images("datasets/vertDetect2d-62_20240226", "datasets/vertDetect2d-62_20240226")
    # rotate_images("datasets/vertDetect2d-62_20240226", "datasets/vertDetect2d-62_20240226")

    ## NT20240408
    # conver_niis2pngs("datasets/NT20240408", "datasets/NT20240408")
    # blackwhite_inverse_image("datasets/NT20240408/case1_AP.bmp", "datasets/NT20240408/case1_AP.bmp")
    # blackwhite_inverse_image("datasets/NT20240408/case1_LAT.bmp", "datasets/NT20240408/case1_LAT.bmp")
    # blackwhite_inverse_image("datasets/NT20240408/case3_AP.png", "datasets/NT20240408/case3_AP.png")
    # blackwhite_inverse_image("datasets/NT20240408/case3_LAT.png", "datasets/NT20240408/case3_LAT.png")
    # crop_image_black_edge("datasets/NT20240408/case2_AP.png", "datasets/NT20240408/case2_AP.png")
    # crop_image_black_edge("datasets/NT20240408/case2_LAT.png", "datasets/NT20240408/case2_LAT.png")

    # TD20240705
    # conver_tiffs2pngs("datasets/TD20240705_LA/fracture", "datasets//TD20240705_LA/fracture")
    # conver_tiffs2pngs("datasets//TD20240705_LA/normal", "datasets//TD20240705_LA/normal")
    # rotate_images("datasets/TD20240705_LA/fracture", "datasets/TD20240705_LA/fracture")
    # rotate_images("datasets/TD20240705_LA/normal", "datasets/TD20240705_LA/normal")
    blackwhite_inverse_images("datasets/TD20240705_LA/fracture", "datasets/TD20240705_LA/fracture")
    blackwhite_inverse_images("datasets/TD20240705_LA/normal", "datasets/TD20240705_LA/normal")
