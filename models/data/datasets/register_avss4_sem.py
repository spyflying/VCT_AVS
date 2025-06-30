import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

logger = logging.getLogger(__name__)

category_labels = {"helicopter": 1, "mynah_bird_singing": 2, "typing_on_computer_keyboard": 3, "playing_violin": 4, "playing_glockenspiel": 5,
                   "playing_piano": 6, "lions_roaring": 7, "baby_laughter": 8, "male_speech": 9, "lawn_mowing": 10, "playing_ukulele": 11,
                   "playing_tabla": 12, "driving_buses": 13, "cap_gun_shooting": 14, "chainsawing_trees": 15, "playing_acoustic_guitar": 16,
                   "cat_meowing": 17, "female_singing": 18, "ambulance_siren": 19, "dog_barking": 20, "horse_clip-clop": 21, "coyote_howling": 22, "race_car": 23,}

def remove_no_use_information(ori_list):
    no_use_information = ["PaxHeader", ".DS_Store", "._.DS_Store"]
    for information in no_use_information:
        if information in ori_list:
            ori_list.remove(information)
    return ori_list
    

def _get_avss4_files(image_dir, gt_dir, audio_dir, pre_mask_dir=None, split="train"):
    only_first_mask = True if split == "train" else False
    files = []
    # scan through the directory
    categories = PathManager.ls(image_dir)
    categories = remove_no_use_information(categories)
    categories = sorted(categories)
    logger.info(f"{len(categories)} categories found in '{image_dir}'.")
    for category in categories:
        assert category in category_labels, "Error category: {}".format(category)
        category_label = category_labels[category]
        category_img_dir = os.path.join(image_dir, category)
        category_gt_dir = os.path.join(gt_dir, category)
        category_pre_mask_dir = os.path.join(pre_mask_dir, category) if pre_mask_dir else None
        category_audio_dir = os.path.join(audio_dir, category)
        videos = PathManager.ls(category_img_dir)
        videos = remove_no_use_information(videos)
        videos = sorted(videos)
        logger.info(f"{len(videos)} videos found in '{category}'.")
        for video in videos:
            video_img_dir = os.path.join(category_img_dir, video)
            video_gt_dir = os.path.join(category_gt_dir, video)
            video_pre_mask_dir = os.path.join(category_pre_mask_dir, video) if category_pre_mask_dir else None
            basenames = PathManager.ls(video_img_dir)
            basenames = remove_no_use_information(basenames)
            image_files = []
            label_files = []
            pre_mask_files = []
            basenames = sorted(basenames)
            audio_file = os.path.join(category_audio_dir, basenames[0][:-6] + ".pkl")
            for num_img, basename in enumerate(basenames):
                image_file = os.path.join(video_img_dir, basename)
                suffix = ".png"
                assert basename.endswith(suffix), basename  # * assert that the file is a png file
                image_files.append(image_file)
                pre_mask_file = os.path.join(video_pre_mask_dir, basename.replace(".png", "_mask_color.png")) if video_pre_mask_dir else None
                pre_mask_files.append(pre_mask_file)
                if (num_img == 0) or (only_first_mask == False):  # * we only have the first frame mask on train.
                    label_file = os.path.join(video_gt_dir, basename)
                    label_files.append(label_file)
                    basename = basename[: -len(suffix)]
            files.append((image_files, label_files, audio_file, pre_mask_files, category_label))

    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0][:-1]:
        assert PathManager.isfile(f[0] if isinstance(f, list) else f), f
    return files


def load_avss4_semantic(img_dir, mask_dir, audio_dir, pre_mask_dir=None, split="train"):
    """
    Args:
        img_dir (str): path to the image directory.
        mask_dir (str): path to the mask directory.
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []
    files = _get_avss4_files(img_dir, mask_dir, audio_dir, pre_mask_dir, split)

    for image_file, label_file, audio_file, pre_mask_file, category_label in files:
        if pre_mask_file[0] is None:
            ret.append(
                {
                    "file_names": image_file,
                    "sem_seg_file_names": label_file,
                    "audio_file_name": audio_file,
                    "category_label": category_label,
                }
            )
        else:
            ret.append(
                {
                    "file_names": image_file,
                    "sem_seg_file_names": label_file,
                    "audio_file_name": audio_file,
                    "pre_mask_file_names": pre_mask_file,
                    "category_label": category_label,
                }
            )
    assert PathManager.isfile(ret[0]["sem_seg_file_names"][0]), ret[0]["sem_seg_file_names"][0]
    return ret


def register_avss4_semantic(root):
    root_224 = os.path.join(root, "s4_data")
    root_384 = os.path.join(root, 's4_data_384')
    for name, dirname in [("train", "train"), ("val", "val"), ("test", "test")]:
        image_dir = os.path.join(root_384, "visual_frames_384", dirname)
        pre_mask_dir = os.path.join(root_384, "pre_SAM_mask_384", dirname)
        audio_dir = os.path.join(root_224, "audio_log_mel", dirname)
        if name == 'train':
            gt_dir = os.path.join(root_384, "gt_masks_384", dirname)
        else: # val and test
            gt_dir = os.path.join(root_384, "gt_masks_224", dirname)

        # pre_mask_dir = os.path.join(root, "pre_SAM_mask_segment_anything", dirname)
        name = f"avss4_sem_seg_{name}"
        DatasetCatalog.register(
            name,
            lambda x=image_dir, y=gt_dir, z=audio_dir, pre_mask_dir=pre_mask_dir, split=dirname: load_avss4_semantic(
                img_dir=x, mask_dir=y, audio_dir=z, pre_mask_dir=pre_mask_dir, split=split
            ),
        )
        MetadataCatalog.get(name).set(
            stuff_classes=["background", "object"],
            evaluator_type="sem_seg",
            ignore_label=255,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
if _root.endswith("/Single-source/"):
    register_avss4_semantic(_root)
