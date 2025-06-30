import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

ss_root = "AVS_dataset/AVSBench_semantic"
s4_root = "AVS_dataset/AVSBench_object/Single-source/s4_data_384"

ss_csv_path = "AVS_dataset/AVSBench_semantic/metadata.csv"
s4_csv_path = "AVS_dataset/AVSBench_object/Single-source/s4_meta_data.csv"

def resize_img(crop_size, img, is_mask=False):
    outsize = crop_size
    if not is_mask:
        img = img.resize((outsize, outsize), Image.BILINEAR)
    else:
        img = img.resize((outsize, outsize), Image.NEAREST)
    return img


if __name__ == '__main__':
    img_384_root = os.path.join(s4_root, 'visual_frames_384')
    gt_224_root = os.path.join(s4_root, 'gt_masks_224')
    gt_384_root = os.path.join(s4_root, 'gt_masks_384')
    pre_mask_384_root = os.path.join(s4_root, 'pre_SAM_mask_384')
    os.makedirs(img_384_root, exist_ok=True)
    os.makedirs(gt_224_root, exist_ok=True)
    os.makedirs(gt_384_root, exist_ok=True)
    os.makedirs(pre_mask_384_root, exist_ok=True)

    df_avss = pd.read_csv(ss_csv_path, sep=',')
    df_v1s = df_avss[df_avss['label'] == 'v1s']
    df_s4 = pd.read_csv(s4_csv_path, sep=',')

    for index in range(len(df_v1s)):
        print("index:", index)
        df_one_video = df_v1s.iloc[index]
        ori_name, video_name, split = df_one_video['vid'], df_one_video['uid'], df_one_video['split']

        filtered_row = df_s4[df_s4['name'] == ori_name]
        if not filtered_row.empty:
            category = filtered_row.iloc[0]['category']
        else:
            raise ValueError(f"No entry found with name '{ori_name}'")
        
        # frames
        T = 5
        img_dir = os.path.join(ss_root, 'v1s', video_name, 'frames')
        img_384_dir = os.path.join(img_384_root, split, category, ori_name)
        os.makedirs(img_384_dir, exist_ok=True)
        for i in range(T):
            img_path = os.path.join(img_dir, f'{i}.jpg')
            img_384_path = os.path.join(img_384_dir, f'{ori_name}_{i + 1}.png')
            img = Image.open(img_path)
            img_384 = resize_img(384, img, is_mask=False)
            img_384.save(img_384_path)
        
        # pre mask
        pre_mask_dir = os.path.join(ss_root, 'pre_SAM_mask/AVSBench_semantic/v1s', video_name, 'frames')
        pre_mask_384_dir = os.path.join(pre_mask_384_root, split, category, ori_name)
        os.makedirs(pre_mask_384_dir, exist_ok=True)
        for i in range(T):
            pre_mask_path = os.path.join(pre_mask_dir, f'{i}_mask_color.png')
            pre_mask_384_path = os.path.join(pre_mask_384_dir, f'{ori_name}_{i + 1}_mask_color.png')
            pre_mask = Image.open(pre_mask_path)
            pre_mask_384 = resize_img(384, pre_mask, is_mask=True)
            pre_mask_384.save(pre_mask_384_path)
        
        # gt-224
        T = 5 if split != 'train' else 1
        gt_dir = os.path.join(ss_root, 'v1s', video_name, 'labels_rgb')
        gt_224_dir = os.path.join(gt_224_root, split, category, ori_name)
        os.makedirs(gt_224_dir, exist_ok=True)
        for i in range(T):
            gt_path = os.path.join(gt_dir, f'{i}.png')
            gt_224_path = os.path.join(gt_224_dir, f'{ori_name}_{i + 1}.png')
            gt = Image.open(gt_path).convert('L')
            gt_array = np.array(gt)
            gt_array[gt_array != 0] = 255
            gt = Image.fromarray(gt_array)
            gt_224 = resize_img(224, gt, is_mask=True)
            gt_224.save(gt_224_path)

        # gt-384
        gt_dir = os.path.join(ss_root, 'v1s', video_name, 'labels_rgb')
        gt_384_dir = os.path.join(gt_384_root, split, category, ori_name)
        os.makedirs(gt_384_dir, exist_ok=True)
        for i in range(T):
            gt_path = os.path.join(gt_dir, f'{i}.png')
            gt_384_path = os.path.join(gt_384_dir, f'{ori_name}_{i + 1}.png')
            gt = Image.open(gt_path).convert('L')
            gt_array = np.array(gt)
            gt_array[gt_array != 0] = 255
            gt = Image.fromarray(gt_array)
            gt_384 = resize_img(384, gt, is_mask=True)
            gt_384.save(gt_384_path)
