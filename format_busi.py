import cv2
import torch
import numpy as np
from tqdm import tqdm

from pathlib import Path
import random


def format_one(mask_path, save_dir):
    """format one image-mask pair

    Args:
        mask_path (str): mask path
        save_dir (str): save directory

    Returns:
        None

    """
    base_name = Path(mask_path).stem.split('_')[0]
    image_path = '%s/%s.png' % (mask_path.parent, base_name)
    label_name = Path(mask_path).parent.stem

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = mask // 255

    if label_name == 'normal':
        label = 0
    elif label_name == 'benign':
        label = 1
    elif label_name == 'malignant':
        label = 2
    else:
        raise ValueError('Unknown label: %s' % label_name)

    Path('%s/images/' % save_dir).mkdir(parents=True, exist_ok=True)
    Path('%s/labels/' % save_dir).mkdir(parents=True, exist_ok=True)

    cv2.imwrite('%s/images/%s.png' % (save_dir, base_name), image)
    torch.save({'mask': mask.astype(np.int16), 'label': label, 'label_name': label_name},
               '%s/labels/%s.pth' % (save_dir, base_name))

    return


if __name__ == '__main__':

    data_dir = '/mnt/d/MyFiles/Research/ImageClassification/data/raw/Dataset_BUSI_with_GT/'
    save_dir = '/mnt/d/MyFiles/Research/ImageClassification/data/formatted/busi/'

    tr_frac = 0.8

    mask_paths = list(Path(data_dir).rglob('*_mask.png'))
    random.shuffle(mask_paths)

    total_len = len(mask_paths)
    tr_len = int(tr_frac * total_len)
    vl_len = total_len - tr_len
    print('Total: %s, Train: %s, Val: %s' % (total_len, tr_len, vl_len))

    for mask_path in tqdm(mask_paths):
        format_one(mask_path, '%s/all/' % save_dir)
    for mask_path in tqdm(mask_paths[:tr_len]):
        format_one(mask_path, '%s/tr/' % save_dir)
    for mask_path in tqdm(mask_paths[tr_len:]):
        format_one(mask_path, '%s/vl/' % save_dir)

    print('Done!')
