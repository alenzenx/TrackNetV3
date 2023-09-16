import os
import cv2
import json
import parse
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from dataset import Badminton_Dataset
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, default='TrackNetV2/model_best.pt')
parser.add_argument('--frame_dir', type=str, default='')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--tolerance', type=float, default=4)
parser.add_argument('--output_mode', type=str, default='frame', choices=['frame', 'heatmap', 'both'])
parser.add_argument('--save_dir', type=str, default='pred_result')
args = parser.parse_args()

model_file = args.model_file
frame_dir = args.frame_dir
batch_size = args.batch_size
tolerance = args.tolerance
output_mode = args.output_mode
save_dir = args.save_dir

checkpoint = torch.load(model_file)
param_dict = checkpoint['param_dict']
model_name = param_dict['model_name']
num_frame = param_dict['num_frame']
input_type = param_dict['input_type']

_, match_id, rally_id = parse.parse('{}/match{}/frame/{}', frame_dir)
output_file = f'{save_dir}/m{match_id}_v{rally_id}.mp4'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
 
# Load dataset
dataset = Badminton_Dataset(frame_dir=frame_dir, mode=input_type, num_frame=num_frame, slideing_step=num_frame)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size, drop_last=False)

# Load model
model = get_model(model_name, num_frame, input_type).cuda()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Video output configuration
FPS = 5
file_names = dataset.frame_files
h, w, _ = cv2.imread(file_names[0][0]).shape
ratio = h / HEIGHT
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
if output_mode == 'both':
    out = cv2.VideoWriter(output_file, fourcc, FPS, (w*2, h))
else:
    out = cv2.VideoWriter(output_file, fourcc, FPS, (w, h))

data_prob = tqdm(data_loader)
for step, (i, x, y, c) in enumerate(data_prob):
    x = x.float().cuda() #(N, 3*F, H, W)
    with torch.no_grad():
        y_pred = model(x)
    x, y = x.detach().cpu().numpy(), y.numpy()
    x = frame_first_RGB(x, input_type)
    y_pred = y_pred.detach().cpu().numpy()
    h_pred = y_pred > 0.5
    y_pred = y_pred * 255.
    h_pred = h_pred * 255.
    y_pred = y_pred.astype('uint8')
    h_pred = h_pred.astype('uint8')
    
    for b in range(y.shape[0]):
        for f in range(num_frame):
            cx_pred, cy_pred = get_object_center(h_pred[b][f])
            cx_pred, cy_pred = int(ratio*cx_pred), int(ratio*cy_pred)
            cx, cy = int(ratio*c[b][f][0]), int(ratio*c[b][f][1])
            pred_type = get_pred_type(cx_pred, cy_pred, cx, cy, ratio*tolerance)
            if output_mode == 'frame':
                file_path = file_names[step*batch_size + b][f]
                img = cv2.imread(file_path)
                if cx_pred != 0 or cy_pred != 0:
                    cv2.circle(img, (cx_pred, cy_pred), 5, (0, 0, 255), -1)
                if cx != 0 or cy != 0:
                    cv2.circle(img, (cx, cy), 4, (218, 255, 51), -1)
                cv2.putText(img, pred_type, (40, 80), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
            elif output_mode == 'heatmap':
                img = np.repeat(h_pred[b][f][:, :, None], 3, axis=2)
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                if cx_pred != 0 or cy_pred != 0:
                    cv2.circle(img, (cx_pred, cy_pred), 5, (0, 0, 255), -1)
                if cx != 0 or cy != 0:
                    cv2.circle(img, (cx, cy), 4, (218, 255, 51), -1)
                cv2.putText(img, pred_type, (40, 80), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
            else: 
                # both
                file_path = file_names[step*batch_size + b][f]
                _, _, frame_id = parse.parse('{}/frame/{}/{}.png', file_path)
                frame = cv2.imread(file_path)
                heatmap = np.repeat(y_pred[b][f][:, :, None], 3, axis=2)
                heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_AREA)
                
                if cx_pred != 0 or cy_pred != 0:
                    cv2.circle(frame, (cx_pred, cy_pred), 3, (0, 0, 255), -1)
                    cv2.circle(heatmap, (cx_pred, cy_pred), 2, (0, 0, 255), -1)
                if cx != 0 or cy != 0:
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
                    cv2.circle(heatmap, (cx, cy), 2, (0, 255, 0), -1)
                cv2.putText(frame, frame_id, (40, 80), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(heatmap, pred_type, (40, 80), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
                img = cv2.hconcat([frame, heatmap])
            out.write(img)
    data_prob.set_description(f'Evaluation')

out.release()