import sys
from face_alignment import FaceAlignment, LandmarksType
from os import listdir, path
import subprocess
import numpy as np
import cv2
import pickle
import os
import json
import torch
from tqdm import tqdm

# Try to import mmpose, if fails use fallback
try:
    from mmpose.apis import inference_topdown, init_model
    from mmpose.structures import merge_data_samples

    MMPOSE_AVAILABLE = True
    print("mmpose available, loading pose model...")
    # initialize the mmpose model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_file = (
        "./musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py"
    )
    checkpoint_file = "./models/dwpose/dw-ll_ucoco_384.pth"
    model = init_model(config_file, checkpoint_file, device=device)
except ImportError as e:
    print(f"mmpose not available ({e}), using face detection only mode")
    MMPOSE_AVAILABLE = False
    model = None

# initialize the face detection model
face_device = "cuda" if torch.cuda.is_available() else "cpu"
fa = FaceAlignment(LandmarksType.TWO_D, flip_input=False, device=face_device)

# maker if the bbox is not sufficient
coord_placeholder = (0.0, 0.0, 0.0, 0.0)


def _sanitize_bbox(x1, y1, x2, y2, frame_shape):
    h, w = frame_shape[:2]
    x1 = int(max(0, min(round(x1), w - 1)))
    y1 = int(max(0, min(round(y1), h - 1)))
    x2 = int(max(0, min(round(x2), w)))
    y2 = int(max(0, min(round(y2), h)))
    if x2 <= x1 or y2 <= y1:
        return coord_placeholder
    return (x1, y1, x2, y2)


def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized


def read_imgs(img_list):
    frames = []
    print("reading images...")
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def get_bbox_range(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [
        frames[i : i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)
    ]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print(
            "get key_landmark and face bounding boxes with the bbox_shift:",
            upperbondrange,
        )
    else:
        print("get key_landmark and face bounding boxes with the default value")
    average_range_minus = []
    average_range_plus = []
    for fb in tqdm(batches):
        if MMPOSE_AVAILABLE and model is not None:
            results = inference_topdown(model, np.asarray(fb)[0])
            results = merge_data_samples(results)
            keypoints = results.pred_instances.keypoints
            face_land_mark = keypoints[0][23:91]
            face_land_mark = face_land_mark.astype(np.int32)
        else:
            face_land_mark = None

        # get bounding boxes by face detection using face_alignment
        faces = fa.face_detector.detect_from_image(fb[0])
        if len(faces) == 0:
            bbox = [None]
        else:
            bbox = [np.array(faces[0])]

        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None:  # no face in the image
                coords_list += [coord_placeholder]
                continue

            if face_land_mark is not None:
                half_face_coord = face_land_mark[
                    29
                ]  # np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
                range_minus = (face_land_mark[30] - face_land_mark[29])[1]
                range_plus = (face_land_mark[29] - face_land_mark[28])[1]
                average_range_minus.append(range_minus)
                average_range_plus.append(range_plus)
                if upperbondrange != 0:
                    half_face_coord[1] = (
                        upperbondrange + half_face_coord[1]
                    )  # 手动调整  + 向下（偏29）  - 向上（偏28）
            else:
                average_range_minus.append(20)
                average_range_plus.append(20)

    if MMPOSE_AVAILABLE and model is not None and average_range_minus:
        text_range = f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}"
    else:
        text_range = f"Total frame:「{len(frames)}」 (Using face detection only mode)"
    return text_range


def get_landmark_and_bbox(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [
        frames[i : i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)
    ]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print(
            "get key_landmark and face bounding boxes with the bbox_shift:",
            upperbondrange,
        )
    else:
        print("get key_landmark and face bounding boxes with the default value")
    average_range_minus = []
    average_range_plus = []
    for fb in tqdm(batches):
        if MMPOSE_AVAILABLE and model is not None:
            results = inference_topdown(model, np.asarray(fb)[0])
            results = merge_data_samples(results)
            keypoints = results.pred_instances.keypoints
            face_land_mark = keypoints[0][23:91]
            face_land_mark = face_land_mark.astype(np.int32)
        else:
            face_land_mark = None

        # get bounding boxes by face detection using face_alignment
        faces = fa.face_detector.detect_from_image(fb[0])
        if len(faces) == 0:
            bbox = [None]
        else:
            bbox = [np.array(faces[0])]

        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None:  # no face in the image
                coords_list += [coord_placeholder]
                continue

            if face_land_mark is not None:
                half_face_coord = face_land_mark[
                    29
                ]  # np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
                range_minus = (face_land_mark[30] - face_land_mark[29])[1]
                range_plus = (face_land_mark[29] - face_land_mark[28])[1]
                average_range_minus.append(range_minus)
                average_range_plus.append(range_plus)
                if upperbondrange != 0:
                    half_face_coord[1] = (
                        upperbondrange + half_face_coord[1]
                    )  # 手动调整  + 向下（偏29）  - 向上（偏28）
                half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
                min_upper_bond = 0
                upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)

                f_landmark = (
                    np.min(face_land_mark[:, 0]),
                    int(upper_bond),
                    np.max(face_land_mark[:, 0]),
                    np.max(face_land_mark[:, 1]),
                )
                x1, y1, x2, y2 = f_landmark

                if (
                    y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0
                ):  # if the landmark bbox is not suitable, reuse the bbox
                    safe_bbox = _sanitize_bbox(f[0], f[1], f[2], f[3], fb[0].shape)
                    coords_list += [safe_bbox]
                    w, h = f[2] - f[0], f[3] - f[1]
                    print("error bbox:", f)
                else:
                    safe_bbox = _sanitize_bbox(x1, y1, x2, y2, fb[0].shape)
                    coords_list += [safe_bbox]
            else:
                # Fallback: just use face detection bounding box with some margin
                x1, y1, x2, y2 = f[:4]  # S3FD returns [x1,y1,x2,y2,score]
                # Add some margin around the face
                h = y2 - y1
                margin = int(h * 0.3)
                y1_new = max(0, y1 - margin)
                y2_new = y2 + margin // 2
                safe_bbox = _sanitize_bbox(x1, y1_new, x2, y2_new, fb[0].shape)
                coords_list += [safe_bbox]

    print(
        "********************************************bbox_shift parameter adjustment**********************************************************"
    )
    if MMPOSE_AVAILABLE and model is not None and average_range_minus:
        print(
            f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}"
        )
    else:
        print(
            f"Total frame:「{len(frames)}」 (Using face detection only mode - install mmcv/mmcv-lite for full functionality)"
        )
    print(
        "*************************************************************************************************************************************"
    )
    return coords_list, frames


if __name__ == "__main__":
    img_list = [
        "./results/lyria/00000.png",
        "./results/lyria/00001.png",
        "./results/lyria/00002.png",
        "./results/lyria/00003.png",
    ]
    crop_coord_path = "./coord_face.pkl"
    coords_list, full_frames = get_landmark_and_bbox(img_list)
    with open(crop_coord_path, "wb") as f:
        pickle.dump(coords_list, f)

    for bbox, frame in zip(coords_list, full_frames):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        print("Cropped shape", crop_frame.shape)

        # cv2.imwrite(path.join(save_dir, '{}.png'.format(i)),full_frames[i][0][y1:y2, x1:x2])
    print(coords_list)
