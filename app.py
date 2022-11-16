from flask import Flask, jsonify, request
import os
import cv2
import json
import glob
import math
import natsort
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data.dataset import Dataset
import torch
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state

app = Flask(__name__)

def make_frame(video_in):
    
    video = cv2.VideoCapture(video_in)

    frame_list = []

    while(video.isOpened()):
        ret, image = video.read()
        frame_list.append(image)
        if len(frame_list) == int(video.get(cv2.CAP_PROP_FRAME_COUNT)): break
    
    video.release()
    
    return frame_list

class CocoValDataset(Dataset):
    def __init__(self, vid):
        super().__init__()
        self._video = vid

    def __getitem__(self, idx):
        return {
            'img': self._video[idx],
        }

    def __len__(self):
        return len(self._video)
    
def run_coco_eval(gt_file_path, dt_file_path):
    annotation_type = 'keypoints'
    print('Running test for {} results.'.format(annotation_type))

    coco_gt = COCO(gt_file_path)
    coco_dt = coco_gt.loadRes(dt_file_path)

    result = COCOeval(coco_gt, coco_dt, annotation_type)
    result.evaluate()
    result.accumulate()
    result.summarize()


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img

def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad

def convert_to_coco_format(pose_entries, all_keypoints):
    coco_keypoints = []
    scores = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        keypoints = [0] * 17 * 3
        to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        person_score = pose_entries[n][-2]
        position_id = -1
        for keypoint_id in pose_entries[n][:-2]:
            position_id += 1
            if position_id == 1:  # no 'neck' in COCO
                continue

            cx, cy, score, visibility = 0, 0, 0, 0  # keypoint not found
            if keypoint_id != -1:
                cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                cx = cx + 0.5
                cy = cy + 0.5
                visibility = 1
            keypoints[to_coco_map[position_id] * 3 + 0] = cx
            keypoints[to_coco_map[position_id] * 3 + 1] = cy
            keypoints[to_coco_map[position_id] * 3 + 2] = visibility
        coco_keypoints.append(keypoints)
        scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))  # -1 for 'neck'
    return coco_keypoints, scores

def infer(net, img, scales, base_height, stride, pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    normed_img = normalize(img, img_mean, img_scale)
    height, width, _ = normed_img.shape
    scales_ratios = [scale * base_height / float(height) for scale in scales]
    avg_heatmaps = np.zeros((height, width, 19), dtype=np.float32)
    avg_pafs = np.zeros((height, width, 38), dtype=np.float32)

    for ratio in scales_ratios:
        scaled_img = cv2.resize(normed_img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        min_dims = [base_height, max(scaled_img.shape[1], base_height)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()#.cuda()
        stages_output = net(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmaps = heatmaps[pad[0]:heatmaps.shape[0] - pad[2], pad[1]:heatmaps.shape[1] - pad[3]:, :]
        heatmaps = cv2.resize(heatmaps, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_heatmaps = avg_heatmaps + heatmaps / len(scales_ratios)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        pafs = pafs[pad[0]:pafs.shape[0] - pad[2], pad[1]:pafs.shape[1] - pad[3], :]
        pafs = cv2.resize(pafs, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_pafs = avg_pafs + pafs / len(scales_ratios)

    return avg_heatmaps, avg_pafs

def evaluate(video, output_name, net, multiscale=False, visualize=False):

    base_height = 368
    scales = [1]
    if multiscale:
        scales = [0.5, 1.0, 1.5, 2.0]
    stride = 8
    
    frame_list = make_frame(video)

    dataset = CocoValDataset(frame_list)
    
    coco_result = []
    keys = []

    for sample in dataset:
        #file_name = sample['file_name']
        img = sample["img"]

        avg_heatmaps, avg_pafs = infer(net, img, scales, base_height, stride)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(18):  # 19th for bg
            total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)

        coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints)
            
        key = [coco_keypoints[0][15], coco_keypoints[0][16], coco_keypoints[0][27], coco_keypoints[0][28]]

        keys.append(key)
        
    return keys
    
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))/np.pi*180

@app.route('/pred', methods=['POST']) 
def test():
    if request.method == 'POST':
        file = request.files['video']
        file.save('src/'+file.filename)
        video = 'src/'+file.filename
        
        net = PoseEstimationWithMobileNet()
        checkpoint = torch.load("src/weight.pth", map_location=torch.device('cpu'))
        load_state(net, checkpoint)

        results = evaluate(video, "testresult",net, True, False)

        vectors = []

        for i in range(len(results)):

            if 0 in results:
                results = [0, 0, 0, 0]

            x = -(results[i][2] - results[i][0])
            y = results[i][3] - results[i][1]

            if x==0 and y==0:
                x = 0.00001
                y = 0.00001

            vectors.append(unit_vector((y,x)))

        arctanres = []

        for vector in vectors:

            rad = np.arctan2(vector[1], vector[0])
            #angle = angle_between((1,0),vector)/np.pi*180
            if rad < 0:
                if -1.57 < rad <= 0:
                    rad = 0
                else:
                    rad += 2*3.14

            arctan_angle = rad/np.pi*180
            #print("angle: ", angle)
            #print("arctan2: ", arctan_angle)

            arctanres.append(arctan_angle)
                        
        max_angle = np.max(arctanres)
        frame_count = np.argmax(arctanres)
        speed = max_angle*((1/30*(frame_count)+1))
                        
        return jsonify({"rom:":max_angle, "speed":speed})

if __name__ == '__main__':
    app.run()
