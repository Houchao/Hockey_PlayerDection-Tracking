from math import ceil
import cv2
import numpy as np
import matplotlib.pyplot as plt
""" red = [(0, 30), (331, 360)]
    yellow = (31, 90)
    green = (91, 150)
    cyan = (151, 210)
    blue = (211, 270)
    purple = (270, 330)"""


# colors are half their normal ranges
def get_hist_ranges(h_counts):
    red = [(0, 15), (166, 180)]
    yellow = (15, 45)
    green = (46, 75)
    cyan = (76, 105)
    blue = (106, 135)
    purple = (136, 165)
    clr_list = [yellow, green, cyan, blue, purple]
    # Add the bins within this range
    bins = np.zeros((1, 6))
    # hardcode red
    r_sum = np.sum(h_counts[red[0][0]:red[0][1]]) + np.sum(
        h_counts[red[1][0]:red[1][1]])
    bins[0, 0] = r_sum
    for i, clr_range in enumerate(clr_list):
        start, stop = clr_range
        bins[0, i + 1] = np.sum(h_counts[start:stop])

    return bins


def plot_img_and_hist(img, window, counts, bins):
    fig = plt.figure(figsize=(18, 10))
    rows = 1
    cols = 3
    if window is not None:
        fig.add_subplot(rows, cols, 1)
        # plt.imshow(cv2.cvtColor(window, cv2.COLOR_BGR2RGB))
        plt.imshow(cv2.cvtColor(window, cv2.COLOR_HSV2RGB))
        plt.title("Window Image")
    if isinstance(bins, range):
        fig.add_subplot(rows, cols, 2)
        plt.bar(bins, counts, width=1)
        plt.title("Hue histogram")
        plt.xlim([0, len(bins)])
        # plt.ylim([0, 1])
        plt.ylim([0, 256])
    else:
        fig.add_subplot(rows, cols, 2)
        plt.bar(bins[:-1], counts, width=1)
        plt.title("Hue histogram")
        plt.xlim([0, 180])

    fig.add_subplot(rows, cols, 3)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Full image")
    fig.show()


def plot_img_and_sat_hist(img, window, counts, bins):
    fig = plt.figure(figsize=(10, 7))
    rows = 1
    cols = 3
    if window is not None:
        fig.add_subplot(rows, cols, 1)
        # plt.imshow(cv2.cvtColor(window, cv2.COLOR_BGR2RGB))
        plt.imshow(cv2.cvtColor(window, cv2.COLOR_HSV2RGB))
        plt.title("Window Image")
    if isinstance(bins, range):
        fig.add_subplot(rows, cols, 2)
        plt.bar(bins, counts, width=1)
        plt.title("Sat histogram")
        plt.xlim([0, len(bins)])
        plt.ylim([0, 255])
    else:
        fig.add_subplot(rows, cols, 2)
        plt.bar(bins[:-1], counts, width=1)
        plt.title("Sat histogram")
        plt.xlim([0, 255])

    fig.add_subplot(rows, cols, 3)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Full image")
    fig.show()


def plot_frame_hists(window, iteration, figure, counts, bins):
    first_pos = iteration
    figure.add_subplot(5, 8, first_pos)
    plt.imshow(cv2.cvtColor(window, cv2.COLOR_BGR2RGB))
    # plt.imshow(cv2.cvtColor(window, cv2.COLOR_HSV2RGB))
    plt.title("Image")
    figure.add_subplot(5, 8, iteration + 1)
    plt.bar(bins, counts, width=1)
    plt.title("Hue hist")
    plt.xlim([0, len(bins)])
    plt.ylim([0, 5])
