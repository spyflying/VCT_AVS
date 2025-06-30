import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

ss_root = "AVS_dataset/AVSBench_semantic"

ss_csv_path = "AVS_dataset/AVSBench_semantic/metadata.csv"

def resize_img(crop_size, img, is_mask=False):
    outsize = crop_size
    if not is_mask:
        img = img.resize((outsize, outsize), Image.BILINEAR)
    else:
        img = img.resize((outsize, outsize), Image.NEAREST)
    return img


if __name__ == '__main__':
    df_avss = pd.read_csv(ss_csv_path, sep=',')
    df_avss = df_avss[df_avss['split'] != 'train'] # only for val and test

    for index in range(len(df_avss)):
        print("index:", index)
        df_one_video = df_avss.iloc[index]
        video_name, split, label = df_one_video['uid'], df_one_video['split'], df_one_video['label']

        video_dir = os.path.join(ss_root, label, video_name)

        # image
        T = 10 if label == 'v2' else 5
        img_dir = os.path.join(video_dir, 'frames')
        img_384_dir = os.path.join(video_dir, 'processed_frames_384')
        os.makedirs(img_384_dir, exist_ok=True)
        for i in range(T):
            img_path = os.path.join(img_dir, f'{i}.jpg')
            img_384_path = os.path.join(img_384_dir, f'{i}.jpg')

            img = Image.open(img_path)
            img_384 = resize_img(384, img, is_mask=False)
            img_384.save(img_384_path)
        
        # pre mask
        T = 10 if label == 'v2' else 5
        pre_mask_dir = os.path.join(ss_root, 'pre_SAM_mask/AVSBench_semantic', label, video_name, 'frames')
        pre_mask_384_dir = os.path.join(ss_root, 'pre_SAM_mask/AVSBench_semantic', label, video_name, 'processed_frames_384')
        os.makedirs(pre_mask_384_dir, exist_ok=True)
        for i in range(T):
            pre_mask_path = os.path.join(pre_mask_dir, f'{i}_mask_color.png')
            pre_mask_384_path = os.path.join(pre_mask_384_dir, f'{i}_mask_color.png')

            pre_mask = Image.open(pre_mask_path)
            pre_mask_384 = resize_img(384, pre_mask, is_mask=True)
            pre_mask_384.save(pre_mask_384_path)
        
        # gt
        T = 10 if label == 'v2' else 5 # only for val and test, so v1s have 5 gt
        gt_dir = os.path.join(video_dir, 'labels_semantic') 
        gt_384_dir = os.path.join(video_dir, 'processed_labels_semantic_384')
        os.makedirs(gt_384_dir, exist_ok=True)
        for i in range(T):
            gt_path = os.path.join(gt_dir, f'{i}.png')
            gt_384_path = os.path.join(gt_384_dir, f'{i}.png')

            gt = Image.open(gt_path)
            gt_384 = resize_img(384, gt, is_mask=True)
            gt_384.save(gt_384_path)

        # rgb gt
        T = 10 if label == 'v2' else 5 # only for val and test, so v1s have 5 gt
        rgb_gt_dir = os.path.join(video_dir, 'labels_rgb') 
        rgb_gt_384_dir = os.path.join(video_dir, 'processed_labels_rgb_384')
        os.makedirs(rgb_gt_384_dir, exist_ok=True)
        for i in range(T):
            rgb_gt_path = os.path.join(rgb_gt_dir, f'{i}.png')
            rgb_gt_384_path = os.path.join(rgb_gt_384_dir, f'{i}.png')

            rgb_gt = Image.open(rgb_gt_path)
            rgb_gt_384 = resize_img(384, rgb_gt, is_mask=True)
            rgb_gt_384.save(rgb_gt_384_path)
