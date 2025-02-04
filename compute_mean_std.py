import cv2
import numpy as np

from pathlib import Path


if __name__ == '__main__':

    image_dir = '/mnt/d/MyFiles/Research/ImageClassification/data/formatted/busi/tr/images/'
    save_file = '/mnt/d/MyFiles/Research/ImageClassification/mean_std.csv'
    images = [cv2.imread(str(image_path))[..., [2, 1, 0]].reshape((-1, 3)) for image_path in Path(image_dir).glob('*.png')]
    images = np.concatenate(images, axis=0) / 255

    means = images.mean(axis=0)
    stds = images.std(axis=0)

    print('mean: %s' % means)
    print('std:  %s' % stds)

    
    np.savetxt(save_file, np.stack([means, stds]), fmt='%.6f', delimiter=',')

    print('Done!')
