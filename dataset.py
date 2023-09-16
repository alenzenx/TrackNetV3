import os
import cv2
import parse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
import tensorflow as tf

from utils import *


class Badminton_Dataset(Dataset):
    def __init__(self, root_dir=data_dir, split='train', mode='2d', num_frame=3, slideing_step=1, frame_dir=None, debug=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.HEIGHT = 288
        self.WIDTH = 512

        self.mag = 1
        self.sigma = 2.5

        self.root_dir = root_dir
        self.split = split
        self.mode = mode
        self.num_frame = num_frame
        self.slideing_step = slideing_step

        if not os.path.exists(os.path.join(self.root_dir, f'f{self.num_frame}_s{self.slideing_step}_{self.split}.npz')):
            self._gen_frame_files()
        data_dict = np.load(os.path.join(self.root_dir, f'f{self.num_frame}_s{self.slideing_step}_{self.split}.npz'))
        
        if debug:
            num_debug = 256
            self.frame_files = data_dict['filename'][:num_debug] # (N, 3)
            self.coordinates = data_dict['coordinates'][:num_debug] # (N, 3, 2)
            self.visibility = data_dict['visibility'][:num_debug] # (N, 3)
        elif frame_dir:
            self.frame_files, self.coordinates, self.visibility = self._gen_frame_unit(frame_dir)
        else:
            self.frame_files = data_dict['filename'] # (N, 3)
            self.coordinates = data_dict['coordinates'] # (N, 3, 2)
            self.visibility = data_dict['visibility'] # (N, 3)

    def _get_rally_dirs(self):
        match_dirs = list_dirs(os.path.join(self.root_dir, self.split))
        match_dirs = sorted(match_dirs, key=lambda s: int(s.split('match')[-1]))
        rally_dirs = []
        for match_dir in match_dirs:
            rally_dir = list_dirs(os.path.join(match_dir, 'frame'))
            rally_dirs.extend(rally_dir)

        # print(rally_dirs)
        #更改
        return rally_dirs

    def _gen_frame_files(self):
        rally_dirs = self._get_rally_dirs()
        frame_files = np.array([]).reshape(0, self.num_frame)
        coordinates = np.array([], dtype=np.float32).reshape(0, self.num_frame, 2)
        visibility = np.array([], dtype=np.float32).reshape(0, self.num_frame)

        # Generate input sequences from each rally
        for rally_dir in tqdm(rally_dirs):
            
            match_dir, rally_id = parse.parse('{}/frame/{}', rally_dir)
            
            csv_file = os.path.join(match_dir, 'csv', f'{rally_id}_ball.csv')
            try:
                label_df = pd.read_csv(csv_file, encoding='utf8').sort_values(by='Frame').fillna(0)
            except:
                print(f'Label file {rally_id}_ball.csv not found.')
                continue
            
            frame_file = np.array([os.path.join(rally_dir, f'{f_id}.png') for f_id in label_df['Frame']])
            x, y, vis = np.array(label_df['X']), np.array(label_df['Y']), np.array(label_df['Visibility'])
            assert len(frame_file) == len(x) == len(y) == len(vis)

            # Sliding on the frame sequence
            for i in range(0, len(frame_file)-self.num_frame, self.slideing_step):
                tmp_frames, tmp_coor, tmp_vis = [], [], []
                # Construct a single input sequence
                for f in range(self.num_frame):
                    if os.path.exists(frame_file[i+f]):
                        tmp_frames.append(frame_file[i+f])
                        tmp_coor.append((x[i+f], y[i+f]))
                        tmp_vis.append(vis[i+f])
                    else:
                        break
                    
                if len(tmp_frames) == self.num_frame:
                    assert len(tmp_frames) == len(tmp_coor) == len(tmp_vis)
                    frame_files = np.concatenate((frame_files, [tmp_frames]), axis=0)
                    coordinates = np.concatenate((coordinates, [tmp_coor]), axis=0)
                    visibility = np.concatenate((visibility, [tmp_vis]), axis=0)
        
        np.savez(os.path.join(self.root_dir, f'f{self.num_frame}_s{self.slideing_step}_{self.split}.npz'), filename=frame_files, coordinates=coordinates, visibility=visibility)

    def _gen_frame_unit(self, frame_dir):
        frame_files = np.array([]).reshape(0, self.num_frame)
        coordinates = np.array([], dtype=np.float32).reshape(0, self.num_frame, 2)
        visibility = np.array([], dtype=np.float32).reshape(0, self.num_frame)
        
        match_dir, rally_id = parse.parse('{}/frame/{}', frame_dir)
        csv_file = f'{match_dir}/csv/{rally_id}_ball.csv'
        label_df = pd.read_csv(csv_file, encoding='utf8').sort_values(by='Frame')
        frame_file = np.array([f'{frame_dir}/{f_id}.png' for f_id in label_df['Frame']])
        x, y, vis = np.array(label_df['X']), np.array(label_df['Y']), np.array(label_df['Visibility'])
        assert len(frame_file) == len(x) == len(y) == len(vis)

        # Sliding on the frame sequence
        for i in range(0, len(frame_file)-self.num_frame, self.slideing_step):
            tmp_frames, tmp_coor, tmp_vis = [], [], []
            # Construct a single input sequence
            for f in range(self.num_frame):
                if os.path.exists(frame_file[i+f]):
                    tmp_frames.append(frame_file[i+f])
                    tmp_coor.append((x[i+f], y[i+f]))
                    tmp_vis.append(vis[i+f])

            # Append the input sequence
            if len(tmp_frames) == self.num_frame:
                assert len(tmp_frames) == len(tmp_coor) == len(tmp_vis)
                frame_files = np.concatenate((frame_files, [tmp_frames]), axis=0)
                coordinates = np.concatenate((coordinates, [tmp_coor]), axis=0)
                visibility = np.concatenate((visibility, [tmp_vis]), axis=0)
        
        return frame_files, coordinates, visibility

    def _get_heatmap(self, cx, cy, visible):
        if not visible:
            return np.zeros((1, self.HEIGHT, self.WIDTH)) if self.mode == '2d' else np.zeros((1, 1, self.HEIGHT, self.WIDTH))
        x, y = np.meshgrid(np.linspace(1, self.WIDTH, self.WIDTH), np.linspace(1, self.HEIGHT, self.HEIGHT))
        heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
        heatmap[heatmap <= self.sigma**2] = 1.
        heatmap[heatmap > self.sigma**2] = 0.
        heatmap = heatmap * self.mag
        return heatmap.reshape(1, self.HEIGHT, self.WIDTH) if self.mode == '2d' else heatmap.reshape(1, 1, self.HEIGHT, self.WIDTH)

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        frame_file = self.frame_files[idx]
        coors = self.coordinates[idx]
        vis = self.visibility[idx]

        # Get the resize scaler
        h, w, _ = cv2.imread(frame_file[0]).shape
        h_ratio, w_ratio = h / self.HEIGHT, w / self.WIDTH

        # Transform the coordinate
        coors[:, 0] = coors[:, 0] / h_ratio
        coors[:, 1] = coors[:, 1] / w_ratio

        if self.mode == '2d':
            frames = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)
            heatmaps = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)

            for i in range(self.num_frame):
                img = tf.keras.utils.load_img(frame_file[i])
                img = tf.keras.utils.img_to_array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                img = np.moveaxis(img, -1, 0)
                frames = np.concatenate((frames, img), axis=0)
                heatmap = self._get_heatmap(int(coors[i][0]), int(coors[i][1]), vis[i])
                heatmaps = np.concatenate((heatmaps, heatmap), axis=0)        
        else:
            frames = np.array([]).reshape(3, 0, self.HEIGHT, self.WIDTH)
            heatmaps = np.array([]).reshape(1, 0, self.HEIGHT, self.WIDTH)

            for i in range(self.num_frame):
                img = tf.keras.utils.load_img(frame_file[i])
                img = tf.keras.utils.img_to_array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                img = np.moveaxis(img, -1, 0) 
                img = img.reshape(3, 1, self.HEIGHT, self.WIDTH)
                frames = np.concatenate((frames, img), axis=1)
                heatmap = self._get_heatmap(int(coors[i][0]), int(coors[i][1]), vis[i])
                heatmaps = np.concatenate((heatmaps, heatmap), axis=1)
        
        frames /= 255.
        return idx, frames, heatmaps, coors
