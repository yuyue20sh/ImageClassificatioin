import torch
import cv2
import numpy as np
import albumentations as A
import pandas as pd

from pathlib import Path

import models


def evaluate(model, image_dir, device, batch_size=64, normalize_args='mean_std.csv'):
    """evaluate model

    Args:
        model (torch.nn.Module): model
        image_dir (str): image directory
        device (str): device

    Returns:
        float: average loss

    """
    files = list(Path(image_dir).glob('*.png'))
    images = [cv2.imread(str(f)) for f in files]
    base_names = [f.stem for f in files]
    print('Loaded %d images' % len(images))

    model.to(device)
    model.eval()

    images = [cv2.imread(str(f)) for f in Path(image_dir).glob('*.png')]
    
    normalize_args = np.loadtxt(normalize_args, delimiter=",")  # mean, std
    transform = A.Compose([
        A.Normalize(mean=normalize_args[0].tolist(), std=normalize_args[1].tolist()),
        A.PadIfNeeded(min_height=512, min_width=512),
        A.RandomCrop(height=512, width=512),
        A.ToTensorV2()
    ])
    images = [transform(image=image)['image'] for image in images]

    n_images = len(images)
    n_batches = n_images // batch_size + 1
    outputs = []
    for i in range(n_batches):
        batch = images[i * batch_size:(i + 1) * batch_size]
        with torch.no_grad():
            outputs.append(model(torch.stack(batch).to(device)).cpu().numpy().argmax(axis=1))

    outputs = np.concatenate(outputs, axis=0).tolist()
    outputs = pd.DataFrame({'image': base_names, 'infer': outputs})

    return outputs


if __name__ == '__main__':

    model_name = 'convnext_tiny'
    n_classes = 3
    weights_path = '/mnt/d/MyFiles/Research/ImageClassification/runs/convnext_tiny_busi/20250204_173113/cp_5.pth'
    device = 'cuda'

    image_dir = '/mnt/d/MyFiles/Research/ImageClassification/data/formatted/busi/vl/images/'
    save_file = 'infer.csv'

    # load the model
    model = models.get_convnext(model_name, n_classes, new_classifier=False)
    model.load_state_dict(torch.load(weights_path, weights_only=False))

    outputs = evaluate(model, image_dir, device)
    outputs.to_csv(save_file, index=False, header=True)
    print(outputs)
    print('Saved to %s' % save_file)

    print('Done!')
