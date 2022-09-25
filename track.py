from collections import deque
import tensorflow as tf
from sklearn.cluster import KMeans
import cluster_hue_sat_funcs as cluster_funcs
import torch.backends.cudnn as cudnn
import torch
import cv2
from pathlib import Path
import time
import shutil
import platform
import argparse
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
    check_imshow,
    xyxy2xywh,
)
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.models.experimental import attempt_load
from yolov5.utils.google_utils import attempt_download
import sys
import os

import numpy as np

import cluster_hue_sat_funcs
import tracking_helpers

sys.path.insert(0, os.path.sep + "yolov5")
#os.environ["Path"]

palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)

group_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def does_box_overlap(box, other_boxes):
    return


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label**2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, color_predictions, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        # color = compute_color_for_labels(id)
        if color_predictions is not None:
            color = group_color[color_predictions[id]]
        else:
            color = (0, 0, 0)
        label = "{}{:d}".format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4),
                      color, -1)
        cv2.putText(
            img,
            label,
            (x1, y1 + t_size[1] + 4),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            [255, 255, 255],
            2,
        )
    return img


def detect(opt):
    (
        out,
        source,
        yolo_weights,
        deep_sort_weights,
        show_vid,
        save_vid,
        save_txt,
        imgsz,
        evaluate,
    ) = (
        opt.output,
        opt.source,
        opt.yolo_weights,
        opt.deep_sort_weights,
        opt.show_vid,
        opt.save_vid,
        opt.save_txt,
        opt.img_size,
        opt.evaluate,
    )
    webcam = (source == "0" or source.startswith("rtsp")
              or source.startswith("http") or source.endswith(".txt"))

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights,
                     repo="mikel-brostrom/Yolov5_DeepSort_Pytorch")
    deepsort = DeepSort(
        cfg.DEEPSORT.REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=True,
    )

    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = (model.module.names if hasattr(model, "module") else model.names
             )  # get class names

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names

    # Run inference
    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split(os.path.sep)[-1].split(".")[0]
    if not args.save_path:
        txt_path = str(Path(out)) + os.path.sep + txt_file_name + ".txt"
    else:
        txt_path = Path(args.save_path, txt_file_name + ".txt")

    cluster_data = np.empty((0, 32))
    kmeans = None
    color_predictions = None
    collect_color_data = False
    id_class_dict = {key: deque([], maxlen=30)
                     for key in list(range(1, 100))
                     }  # Could be better optimized
    txt_lines_to_write = []

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes=opt.classes,
            agnostic=opt.agnostic_nms,
        )
        t2 = time_synchronized()

        # Process detections
        start_process_time = time.perf_counter()
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], "%g: " % i, im0s[i].copy()
            else:
                p, s, im0 = path, "", im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                          im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confss = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                # print("class =", clss, "class.cpu", clss.cpu())
                outputs = deepsort.update(xywhs.cpu(), confss.cpu(),
                                          clss.cpu(), im0)

                if len(outputs) > 0:
                    collect_color_data = True
                    bboxes = outputs[:, :4]
                    identities = outputs[:, -2]

                    hist_data = np.zeros(
                        (bboxes.shape[0], 32)
                    )  # 32 is dependent on window size, number of peaks, etc..
                    hist_data = cluster_funcs.get_histogram_data(
                        im0, bboxes, hist_data)

                mode_in_class_id_dict = None

                if frame_idx < 25 and collect_color_data:
                    cluster_data = np.vstack((cluster_data, hist_data))
                elif frame_idx == 25:
                    kmeans = KMeans(n_clusters=2,
                                    init="k-means++",
                                    random_state=0).fit(cluster_data)
                elif frame_idx > 25:
                    color_predictions = kmeans.predict(hist_data)
                    identities = outputs[:, -2]
                    # TODO Change this to ids in last frame to make indices match

                    for predict_class, id in zip(color_predictions,
                                                 identities):
                        id_class_dict[id].append(predict_class)
                    mode_in_class_id_dict = cluster_hue_sat_funcs.get_counts(
                        identities, id_class_dict)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    draw_boxes(im0, bbox_xyxy, mode_in_class_id_dict,
                               identities)  # TODO uncomment when needed again
                    # to MOT format
                    tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)

                    # Write MOT compliant results to file
                    ids = []
                    if save_txt:
                        for j, (tlwh_bbox,
                                output) in enumerate(zip(tlwh_bboxs, outputs)):
                            bbox_top = tlwh_bbox[0]
                            bbox_left = tlwh_bbox[1]
                            bbox_w = tlwh_bbox[2]
                            bbox_h = tlwh_bbox[3]
                            identity = output[-1]
                            ids.append(identity)
                            if frame_idx <= 25:
                                string_line = ("%g " * 10 + "\n") % (
                                    frame_idx,
                                    identity,
                                    bbox_top,
                                    bbox_left,
                                    bbox_w,
                                    bbox_h,
                                    -1,
                                    -1,
                                    -1,
                                    -1,
                                )
                                txt_lines_to_write.append(string_line)
                            else:
                                classif = cluster_hue_sat_funcs.get_counts(
                                    [identity], id_class_dict)
                                string_line = ("%g " * 10 + "\n") % (
                                    frame_idx,
                                    identity,
                                    bbox_top,
                                    bbox_left,
                                    bbox_w,
                                    bbox_h,
                                    classif[identity],
                                    -1,
                                    -1,
                                    -1,
                                )  # label format
                                txt_lines_to_write.append(string_line)
                            # with open(txt_path, 'a') as f:
                            #     if frame_idx <= 25:
                            #         f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_top,
                            #                                    bbox_left, bbox_w, bbox_h, -1, -1, -1, -1)) # label format
                            #     else:
                            #         classif = cluster_hue_sat_funcs.get_counts([identity], id_class_dict)
                            #         f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_top,
                            #                                        bbox_left, bbox_w, bbox_h, classif[identity], -1, -1,
                            #                                        -1))  # label format
                            # f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_top,
                            #                             bbox_left, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            print(
                f"{s} Done. {t2 - t1 + (time.perf_counter() - start_process_time)}"
            )

            # Stream results
            if show_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord("q"):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += ".mp4"

                    vid_writer = cv2.VideoWriter(
                        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                        (w, h))
                    # Draw frame number in top left
                cv2.putText(
                    im0,
                    f"frame index = {frame_idx}",
                    (100, 200),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    [255, 255, 255],
                    2,
                )
                vid_writer.write(im0)

    if save_txt or save_vid:
        print("Results saved to %s" % os.getcwd() + os.sep + out)
        if platform == "darwin":  # MacOS
            os.system("open " + save_path)

    print("Done. (%.3fs)" % (time.time() - t0))
    with open(txt_path, "a") as f:
        f.writelines(txt_lines_to_write)
    # vid_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yolo_weights",
        type=str,
        default="yolov5/weights/yolov5s.pt",
        help="model.pt path",
    )
    parser.add_argument(
        "--deep_sort_weights",
        type=str,
        default="deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7",
        help="ckpt.t7 path",
    )
    # file/folder, 0 for webcam
    parser.add_argument("--source", type=str, default="0", help="source")
    parser.add_argument("--output",
                        type=str,
                        default="inference/output",
                        help="output folder")  # output folder
    parser.add_argument("--img-size",
                        type=int,
                        default=640,
                        help="inference size (pixels)")
    parser.add_argument("--conf-thres",
                        type=float,
                        default=0.4,
                        help="object confidence threshold")
    parser.add_argument("--iou-thres",
                        type=float,
                        default=0.5,
                        help="IOU threshold for NMS")
    parser.add_argument(
        "--fourcc",
        type=str,
        default="mp4v",
        help="output video codec (verify ffmpeg support)",
    )
    parser.add_argument("--device",
                        default="",
                        help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--show-vid",
                        action="store_true",
                        help="display tracking video results")
    parser.add_argument("--save-vid",
                        action="store_true",
                        help="save video tracking results")
    parser.add_argument("--save-txt",
                        action="store_true",
                        help="save MOT compliant results to *.txt")
    parser.add_argument("--save_path",
                        type=str,
                        help="save MOT compliant results to *.txt")
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 16 17",
    )
    parser.add_argument("--agnostic-nms",
                        action="store_true",
                        help="class-agnostic NMS")
    parser.add_argument("--augment",
                        action="store_true",
                        help="augmented inference")
    parser.add_argument("--evaluate",
                        action="store_true",
                        help="augmented inference")
    parser.add_argument(
        "--config_deepsort",
        type=str,
        default="deep_sort_pytorch/configs/deep_sort.yaml",
    )
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
