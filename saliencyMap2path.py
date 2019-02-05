# coding:utf-8
"""
python2
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import math
import json

import numpy as np


def calc_saliency_of_superpixel(segment_map_file, saliency_map_file, length=5, debug=False):
    """
    input:
        segment_map: 2d-array .npy
        saliency_map: 2d-array .npy
        length: length of predicted scanpath
    output:
        list of fixations
    """
    segment_map = np.load(segment_map_file)
    saliency_map = np.load(saliency_map_file)

    verbose_sm = np.zeros(np.shape(saliency_map))
    saliency_dict = {}
    center_dict = {}

    seg_idx_min = np.min(segment_map)
    seg_idx_max = np.max(segment_map)
    for seg_idx in range(seg_idx_min, seg_idx_max+1):
        pos_y, pos_x = np.where(segment_map==seg_idx)
        saliency_values = saliency_map[pos_y, pos_x]
        saliency_value = np.mean(saliency_values)
        ## store saliency value of superpixel
        verbose_sm[pos_y, pos_x] = saliency_value
        saliency_dict[seg_idx] = saliency_value
        ## calc center position of superpixel
        center_x = math.floor(np.mean(pos_x))
        center_y = math.floor(np.mean(pos_y))
        # print("{}\t{}".format(center_x, center_y))
        center_dict[seg_idx] = (center_y, center_x)
    
    saliency_dict_sorted = sorted(saliency_dict.iteritems(), key=lambda t: t[1], reverse=True)
    
    out_saliency = saliency_dict_sorted[:length]
    out_position = []
    for seg_idx, saliency_value in out_saliency:
        out_position.append(center_dict[seg_idx])

    if debug:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.imshow(saliency_map)
        ax2 = fig.add_subplot(122)
        ax2.imshow(verbose_sm)
        for seg_idx in range(seg_idx_min, seg_idx_max+1):
            y, x = center_dict[seg_idx]
            value = saliency_dict[seg_idx]
            ax2.text(x, y, "{:.2f}".format(value))
        for i, (y, x) in enumerate(out_position, start=1):
            ax2.text(x-2, y-2, i, color='r')
        plt.show()
    
    return out_position


def main(segment_map_root, saliency_map_root, out_file, length):
    scanpath_pred = {}
    count = 0
    for img_name in os.listdir(segment_map_root):
        img_id = img_name.split('.')[0]
        segment_map_file = os.path.join(segment_map_root, img_name)
        saliency_map_file = os.path.join(saliency_map_root, img_name)
        # print("========")
        # print(os.path.exists(segment_map_file))
        # print(os.path.exists(saliency_map_file))
        scanpath = calc_saliency_of_superpixel(segment_map_file=segment_map_file, saliency_map_file=saliency_map_file, length=length)
        scanpath_pred[img_id] = json.dumps(scanpath)
        count += 1
        print(count)

    with open(out_file, 'w') as fh:
        for img_id, scanpath in scanpath_pred:
            fh.write("{}\t{}".format(img_id, scanpath))
        print('done')


if __name__ == "__main__":
    # calc_saliency_of_superpixel(
    #     segment_map_file='/home/lixiang/Desktop/personal-code/dataset/JUDD/Segment/SLIC#300/i05june05_static_street_boston_p1010800.npy',
    #     saliency_map_file='/home/lixiang/Desktop/personal-code/1998-ltti/result_saliencyMap/i05june05_static_street_boston_p1010800.npy',
    #     debug=True
    # )
    length = 5
    main(
        segment_map_root='/home/lixiang/Desktop/personal-code/dataset/JUDD/Segment/SLIC#300',
        saliency_map_root='/home/lixiang/Desktop/personal-code/1998-ltti/result_saliencyMap',
        out_file='/home/lixiang/Desktop/personal-code/1998-ltti/result_scanpath#{}'.format(length),
        length=length
    )