import os
import cv2
import json
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from dataset import Badminton_Dataset
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--video_file', type=str)
parser.add_argument('--model_file', type=str, default='TrackNetV2/model_best.pt')
parser.add_argument('--num_frame', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--save_dir', type=str, default='pred_result')
args = parser.parse_args()

video_file = args.video_file
model_file = args.model_file
num_frame = args.num_frame
batch_size = args.batch_size
save_dir = args.save_dir

video_name = video_file.split('/')[-1][:-4]
video_format = video_file.split('/')[-1][-3:]
out_video_file = f'{save_dir}/{video_name}_pred.{video_format}'
out_csv_file = f'{save_dir}/{video_name}_ball.csv'

checkpoint = torch.load(model_file)
param_dict = checkpoint['param_dict']
model_name = param_dict['model_name']
num_frame = param_dict['num_frame']
input_type = param_dict['input_type']

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
 
# Load model
model = get_model(model_name, num_frame, input_type).cuda()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Video output configuration
if video_format == 'avi':
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
elif video_format == 'mp4':
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
else:
    raise ValueError('Invalid video format.')

# Write csv file head
f = open(out_csv_file, 'w')
f.write('Frame,Visibility,X,Y\n')

# Cap configuration
cap = cv2.VideoCapture(video_file)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
success = True
frame_count = 0
num_final_frame = 0
ratio = h / HEIGHT
out = cv2.VideoWriter(out_video_file, fourcc, fps, (w, h))

while success:
    print(f'Number of sampled frames: {frame_count}')
    # Sample frames to form input sequence
    frame_queue = []
    for _ in range(num_frame*batch_size):
        success, frame = cap.read()
        if not success:
            break
        else:
            frame_count += 1
            frame_queue.append(frame)

    if not frame_queue:
        break
    
    # If mini batch incomplete
    if len(frame_queue) % num_frame != 0:
        frame_queue = []
        # Record the length of remain frames
        num_final_frame = len(frame_queue) +1
        print(num_final_frame)
        # Adjust the sample timestampe of cap
        frame_count = frame_count - num_frame*batch_size
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        # Re-sample mini batch
        for _ in range(num_frame*batch_size):
            success, frame = cap.read()
            if not success:
                break
            else:
                frame_count += 1
                frame_queue.append(frame)
        if len(frame_queue) % num_frame != 0:
            continue
    
    x = get_frame_unit(frame_queue, num_frame)
    
    # Inference
    with torch.no_grad():
        y_pred = model(x.cuda())
    y_pred = y_pred.detach().cpu().numpy()
    h_pred = y_pred > 0.5
    h_pred = h_pred * 255.
    h_pred = h_pred.astype('uint8')
    h_pred = h_pred.reshape(-1, HEIGHT, WIDTH)
    
    for i in range(h_pred.shape[0]):
        if num_final_frame > 0 and i < (num_frame*batch_size - num_final_frame-1):
            print('aaa')
            # Special case of last incomplete mini batch
            # Igore the frame which is already written to the output video
            continue 
        else:
            img = frame_queue[i].copy()
            cx_pred, cy_pred = get_object_center(h_pred[i])
            cx_pred, cy_pred = int(ratio*cx_pred), int(ratio*cy_pred)
            vis = 1 if cx_pred > 0 and cy_pred > 0 else 0
            # Write prediction result
            f.write(f'{frame_count-(num_frame*batch_size)+i},{vis},{cx_pred},{cy_pred}\n')
            # print(frame_count-(num_frame*batch_size)+i)
            if cx_pred != 0 or cy_pred != 0:
                cv2.circle(img, (cx_pred, cy_pred), 5, (0, 0, 255), -1)
            out.write(img)

out.release()
print('Done.')