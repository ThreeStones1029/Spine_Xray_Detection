'''
Description: 
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-16 02:22:36
LastEditors: ShuaiLei
LastEditTime: 2024-04-16 13:52:58
'''
""""by lyuwenyu
"""

import torch
import torch.utils.data

import torchvision
torchvision.disable_beta_transforms_warning()
import numpy as np
import PIL
from PIL import Image, ImageDraw


__all__ = ['show_sample']

def show_sample(sample):
    """for coco dataset/dataloader
    """
    import matplotlib.pyplot as plt
    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes

    image, target = sample
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)

    image = F.convert_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(image, target["boxes"], colors="yellow", width=3)

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()
    fig.show()
    plt.show()


def draw_bbox(image, im_id, catid2name, bboxes, threshold):
    """
    Draw bbox on image
    """
    draw = ImageDraw.Draw(image)

    for dt in np.array(bboxes):
        if im_id != dt['image_id']:
            continue
        catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
        if score < threshold:
            continue
        # draw bbox
        if len(bbox) == 4:
            # draw bbox
            xmin, ymin, w, h = bbox
            xmax = xmin + w
            ymax = ymin + h
            draw.line(
                [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                (xmin, ymin)],
                width=2,
                fill='red')
        else:
            print('the shape of bbox must be [M, 4] or [M, 8]!')

        # draw label
        text = "{} {:.4f}".format(catid2name[catid], score)
        # tw, th = draw.textsize(text)
        left, top, right, bottom = draw.textbbox((0, 0), text)
        tw, th = right - left, bottom - top

        #label框
        draw.rectangle([(xmin + 1, ymin + 1), (xmin + tw + 1, ymin + th + 1 + 10)], fill='white') 
        # draw.rectangle([(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill = color)
        
        # label文字 
        # (xmin + 1, ymin - th)
        draw.text((xmin + 1, ymin + 1), text, fill='red') 
        # draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))
    
    return image

