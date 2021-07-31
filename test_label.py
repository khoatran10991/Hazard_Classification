#!/usr/bin/env python3
import argparse
import sys
import cv2
import numpy as np
import os
from process import HazardClassification
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("TkAgg")

def main(args):
    classifier = HazardClassification(appEfficientNet=args.appEfficientNet,label=args.mapping_file,ckpt=args.model_ckpt)
    fig, ax = plt.subplots(4,5, figsize=(20,10))
    # Loading image"
    fileNames = list(filter(lambda jpg: jpg[-3:].lower() == 'jpg', os.listdir('./images-cropped/')))
    random_imgs = np.random.choice(fileNames, size=20, replace=False)
    for idx, file in enumerate(random_imgs.tolist()):
        img = cv2.imread(os.path.join("images-cropped", file), cv2.IMREAD_UNCHANGED)
        class_res = classifier.run(image=img)
        print('Idx: {}'.format(idx))
        ax[int(idx/5), idx%5].imshow(img)
        ax[int(idx/5), idx%5].axis('off')
        ax[int(idx/5), idx%5].set_title(class_res)
    plt.show()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping_file', help='Path to save label map file.', type=str, default='label_map.pkl')
    parser.add_argument('--appEfficientNet', help='EfficientNetB0, 1, 2, 3, 4, 5, 6, 7', type=int, default=0)
    parser.add_argument('--model_ckpt', help='File name model ckpt h5', type=str, default='model_best_ckpt.h5')

    args = parser.parse_args()
    main(args)