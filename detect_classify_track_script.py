import os
import subprocess
import sys
import logging
import time
from tqdm import tqdm

logging.basicConfig(handlers=[logging.FileHandler(filename=r"C:\Users\Andre\PycharmProjects\HockeyResearch\time_to_create2.txt",
                                                      encoding='utf-8', mode='a+')],
                        format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%F %A %T",
                        level=logging.DEBUG)

dataset_FP = r'D:\Final_dataset\dataset_including_full_shot_clip\data'
# save_FP = r'C:\Users\Andre\PycharmProjects\HockeyResearch\track_class_detect'
save_FP = r'D:\still&bounds_no_ref\frame_clusters&ids'

for file in tqdm(os.listdir(dataset_FP)):
    if file.endswith('.mp4'):
        txt_filename = file.replace('.mp4', '.txt')
        if os.path.isfile(os.path.join(r'D:\still&bounds_no_ref\frame_clusters&ids', txt_filename)):
            print(f"Skipping file {file} because it already exists.")
            continue
        # --yolo_weights 20210430best.pt --source D:\Final_dataset\including_overlap\data\event2105.mp4 --save-vid --device cpu --save-txt
        start = time.perf_counter()
        subprocess.call(
            [sys.executable, r"C:\Users\Andre\PycharmProjects\Yolov5_DeepSort_Pytorch\track.py", "--yolo_weights",
             r'C:\Users\Andre\PycharmProjects\Yolov5_DeepSort_Pytorch\best.pt',
             "--source", os.path.join(dataset_FP, file), '--save-txt', '--save_path', save_FP, "--config_deepsort",
             r'C:\Users\Andre\PycharmProjects\Yolov5_DeepSort_Pytorch\deep_sort_pytorch\configs\deep_sort.yaml'], stdout=open(os.devnull, 'wb')) # removed save vid
        logging.debug(f"Time taken for {file} is {time.perf_counter() - start}")