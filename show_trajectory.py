import os
import cv2

import argparse
import numpy as np
import pandas as pd
from collections import deque
from PIL import Image, ImageDraw
from keras.models import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--video_file', type=str)
parser.add_argument('--csv_file', type=str)
parser.add_argument('--traj_len', type=int, default=8)
parser.add_argument('--save_dir', type=str, default='pred_result')
args = parser.parse_args()

video_file = args.video_file
csv_file = args.csv_file
traj_len = args.traj_len
save_dir = args.save_dir
video_name = video_file.split('/')[-1][:-4]
output_video_file = f'{save_dir}/{video_name}_traj.mp4'

# Read prediction result of the input video
label_df = pd.read_csv(csv_file, encoding='utf8')
frame_id = np.array(label_df['Frame'])
x, y, vis = np.array(label_df['X']), np.array(label_df['Y']), np.array(label_df['Visibility'])
print(f'total frames: {len(frame_id)}')

# For storing trajectory
queue = deque()

# Cap configuration
cap = cv2.VideoCapture(video_file)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
success = True
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_file, fourcc, fps, (w, h))

frame_i = 0
while success:
    success, frame = cap.read()
    if not success:
        break

    # Push ball coordinates for each frame
    if vis[frame_i]:
        if len(queue) >= traj_len:
            queue.pop()
        queue.appendleft([x[frame_i], y[frame_i]])
    else:
        if len(queue) >= traj_len:
            queue.pop()
        queue.appendleft(None)

    # Convert to PIL image for drawing
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   
    img = Image.fromarray(img)

    # Draw ball trajectory
    for i in range(len(queue)):
        if queue[i] is not None:
            draw_x = queue[i][0]
            draw_y = queue[i][1]
            bbox =  (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
            draw = ImageDraw.Draw(img)
            draw.ellipse(bbox, outline ='yellow')
            del draw

    # Convert back to cv2 image and write to output video
    frame =  cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    out.write(frame)
    frame_i += 1

out.release()
cap.release()
print('Done')