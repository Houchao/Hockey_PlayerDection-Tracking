import sys
import time
from math import ceil, floor

import cv2
import numpy as np
import os
import color_space_funcs as funcs
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
import copy
import ast
import argparse
from collections import Counter

# Uncomment this line in order to produce seperate windows for plots
# matplotlib.use('Qt5Agg')

HUE = 0
SATURATION = 1
VALUE = 2


def get_counts(ids, classif_dict):
    id_counts = {key: None for key in ids}
    for id in ids:
        id_counts[id] = Counter(
            classif_dict[id]).most_common(1)[0][0]  #returns the mode
    return id_counts


def get_boxes(path):
    boxes = []
    with open(path, 'r+') as file:
        lines = file.readlines()
        for line in lines:
            box_elements = line.split(',')
            box = (float(box_elements[0]), float(box_elements[1]),
                   float(box_elements[2]), float(box_elements[3]))
            boxes.append(box)

    return boxes


def get_bboxes():
    path_to_boxes = r"/bboxes.txt"
    bboxes_ls = []
    if os.path.isfile(path_to_boxes):
        bboxes_ls = get_boxes(path_to_boxes)
    return bboxes_ls


def threshold_gray(img, thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ind = gray < thresh
    return ind


def get_hsv_hist(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ind = threshold_gray(img, 197)
    h, s, v = hsv[:, :, HUE], hsv[:, :, SATURATION], hsv[:, :, VALUE]
    h_no_ice = h[ind]
    s_no_ice = v[ind]

    hist2d, xbins, ybins = np.histogram2d(
        h_no_ice, s_no_ice, [180, 256], [[0, 180], [0, 256]])  #, normed=True)

    return hsv, hist2d


def crop_to_uniform(img, ratio):
    r, c, channels = img.shape

    img = cv2.resize(img, (24, 48), interpolation=cv2.INTER_AREA)
    rows, cols, dims = img.shape
    quarter_rows = rows / ratio
    quarter_cols = cols / ratio
    cropped_img = img[int(quarter_rows):int(quarter_rows * 3),
                      int(quarter_cols):int(quarter_cols * 3)]
    return cropped_img


def get_bounding_boxes(bbox_path):
    """
    File is in the form: frame_idx, id, bbox_top, bbox left, bbox_w, bbox_h
    :return:
    """
    with open(bbox_path, 'r') as file:
        bbox_list = []
        frame_list = []
        frame_idx = -66
        for line in file.readlines():
            line_ls = line.split()
            new_frame_id = int(line_ls[0])
            if new_frame_id > frame_idx:
                frame_idx = new_frame_id
                if frame_list:
                    bbox_list.append(copy.deepcopy(frame_list))
                    frame_list.clear()
            frame_list.append(line_ls[:6])
        bbox_list.append(frame_list)
    return bbox_list


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield x, y, image[y:y + windowSize[1], x:x + windowSize[0]]


def get_peaks(h_counts, num_peaks):
    ind = np.argpartition(h_counts, -num_peaks)[-num_peaks:]
    sorted_ind = ind[np.argsort((-h_counts)[ind])]
    return sorted_ind


def get_range(h_counts):
    ranges = funcs.get_hist_ranges(h_counts)
    norm = np.linalg.norm(ranges)
    ranges = ranges / norm
    return ranges


def get_sliding_window_hist(img, technique='peaks'):
    peaks_num = 2
    num_peaks = peaks_num
    num_sat_peaks = peaks_num
    # if technique == 'peaks':
    #     num_peaks = 3
    # elif technique == 'range':
    #     num_peaks = 6
    step_size = 6
    windows = sliding_window(img, step_size, (6, 6))
    num_windows = ceil(img.shape[0] / step_size) * ceil(
        img.shape[1] / step_size)
    horizontal_hists = np.zeros(
        (1, num_peaks * num_windows + num_sat_peaks * num_windows))
    start = 0
    stop = num_peaks + num_sat_peaks
    for i, window_info in enumerate(windows):
        window = window_info[2]
        hsv, hist2d = get_hsv_hist(window)
        peaks = get_2d_peaks(num_peaks, hist2d)
        # if technique == 'peaks':
        #     horizontal_hists[0, start:stop] = get_peaks(h_counts, num_peaks)
        # elif technique == 'range':
        #     horizontal_hists[0, start:stop] = get_range(h_counts)

        horizontal_hists[0, start:stop] = peaks
        start += num_peaks + num_sat_peaks
        stop += num_peaks + num_sat_peaks
    return horizontal_hists


def convert_to_ints(*args):
    int_list = []
    for num in args:
        int_list.append(int(num))
    return int_list


def get_2d_peaks(num_peaks, hist2d):
    flat_hist = hist2d.ravel()
    ind = np.argpartition(flat_hist, -num_peaks)[-num_peaks:]
    sorted_ind = ind[np.argsort((-flat_hist)[ind])]
    row_idxs = []
    col_idxs = []
    for idx in sorted_ind:
        row = floor(idx / 256)
        col = (idx % 256)
        row_idxs.append(row)
        col_idxs.append(col)

    idxs = np.hstack((row_idxs, col_idxs)).ravel()

    return idxs


def read_ls_input(tlwh_ls):
    boxes = ast.literal_eval(tlwh_ls)
    boxes = np.array(boxes)
    return boxes


def get_histogram_data(img, tlwh_boxes, arr):
    for i, box in enumerate(tlwh_boxes):

        bbox_top, bbox_left, bbox_h, bbox_w = convert_to_ints(
            box[0], box[1], box[2], box[3])

        imCrop = img[bbox_left:bbox_w, bbox_top:bbox_h]
        uniform_crop = crop_to_uniform(imCrop, 4)

        hist_data = get_sliding_window_hist(uniform_crop)
        arr[i, :] = hist_data
    return arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Creates clips based on shot times from csv')
    parser.add_argument('-tlwh_box',
                        type=str,
                        help='list of bounding boxes in the form of a string')
    args = parser.parse_args()

    tlwh_boxes = read_ls_input(args.tlwh_box)
    get_sliding_window_hist()
