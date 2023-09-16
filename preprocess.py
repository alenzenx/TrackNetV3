import os
import parse
import argparse
from utils import *

for split in ['train', 'test']:
    s_frame_count = 0
    for match_dir in list_dirs(f'{data_dir}/{split}'):
        match_name = match_dir.split('/')[-1]
        video_files = list_dirs(f'{match_dir}/video')
        print(video_files)
        # os.system('pause')
        m_frame_count = 0
        for video_file in video_files:
            generate_frames(video_file)
            print(video_file)
            v_frame_count = get_num_frames(video_file)
            video_name = video_file.split('/')[-1]
            print(f'[{split} / {match_name} / {video_name}]\tvideo frames: {v_frame_count}')
            m_frame_count += v_frame_count

        print(f'[{split} / {match_name}]:\ttotal frames: {m_frame_count}')
        s_frame_count += m_frame_count
    
    print(f'[{split}]:\ttotal frames: {s_frame_count}')