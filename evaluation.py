import os
import json
import parse
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from dataset import Badminton_Dataset
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, default='TrackNetV2/model_best.pt')
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--tolerance', type=float, default=4)
parser.add_argument('--save_dir', type=str, default='output')
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--analyze', action='store_true', default=False)
args = parser.parse_args()

model_file = args.model_file
batch_size = args.batch_size
tolerance = args.tolerance
save_dir = args.save_dir
ckpt_dir = model_file.split('/model')[0]
debug = args.debug
analyze = args.analyze

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

checkpoint = torch.load(model_file)
param_dict = checkpoint['param_dict']
model_name = param_dict['model_name']
num_frame = param_dict['num_frame']
input_type = param_dict['input_type']

if __name__ == '__main__':
    if not analyze:
        # Load dataset
        print(f'Data dir: {data_dir}')
        print(f'Data input type: {input_type}')
        train_dataset = Badminton_Dataset(root_dir=data_dir, split='train', mode=input_type, num_frame=num_frame, slideing_step=1, debug=debug)
        test_dataset = Badminton_Dataset(root_dir=data_dir, split='test', mode=input_type, num_frame=num_frame, slideing_step=1, debug=debug)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        # Load model
        model = get_model(model_name, num_frame, input_type).cuda()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        if not os.path.exists(f'{save_dir}/eval_analysis.json'):
            print(f'Tolerance Value: {tolerance}')
            print('Train evaluation')
            accuracy, precision, recall, TP, TN, FP1, FP2, FN = evaluation(model, train_loader, tolerance, input_type)
            train_res_dict = {'TP':TP, 'TN': TN, 'FP1': FP1, 'FP2': FP2, 'FN': FN}
            
            print('Test evaluation')
            accuracy, precision, recall, TP, TN, FP1, FP2, FN = evaluation(model, test_loader, tolerance, input_type)
            test_res_dict = {'TP':TP, 'TN': TN, 'FP1': FP1, 'FP2': FP2, 'FN': FN}
            
            eval_dict = dict(train=train_res_dict, test=test_res_dict, data_dir=data_dir)
            with open(f'{save_dir}/eval_analysis.json', 'w') as f:
                json.dump(eval_dict, f, indent=2)
        
        if not os.path.exists(f'{save_dir}/eval_res.json'):
            eval_dict = json.load(open(f'{save_dir}/eval_analysis.json'))
            train_res_dict = eval_dict['train']
            TP, TN, FP1, FP2, FN = len(train_res_dict['TP']), len(train_res_dict['TN']), len(train_res_dict['FP1']), len(train_res_dict['FP2']), len(train_res_dict['FN'])
            accuracy, precision, recall = get_metric(TP, TN, FP1, FP2, FN)
            train_res = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'TP':TP, 'TN': TN, 'FP1': FP1, 'FP2': FP2, 'FN': FN}
            test_res_dict = eval_dict['test']
            TP, TN, FP1, FP2, FN = len(test_res_dict['TP']), len(test_res_dict['TN']), len(test_res_dict['FP1']), len(test_res_dict['FP2']), len(test_res_dict['FN'])
            accuracy, precision, recall = get_metric(TP, TN, FP1, FP2, FN)
            test_res = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'TP':TP, 'TN': TN, 'FP1': FP1, 'FP2': FP2, 'FN': FN}
            res_dict = dict(train=train_res, test=test_res)
            with open(f'{save_dir}/eval_res.json', 'w') as f:
                json.dump(res_dict, f, indent=2)
        
        loss_list = checkpoint['loss_list']
        test_acc_dict = checkpoint['test_acc']
        plot_result(loss_list, None, test_acc_dict, num_frame, save_dir, model_name)

    else:
        # Gather statistic from evaluation results
        assert os.path.exists(f'{save_dir}/eval_analysis.json')
        train_data = np.load(f'{data_dir}/f{num_frame}_s1_train.npz')
        test_data = np.load(f'{data_dir}/f{num_frame}_s1_test.npz')
        eval_dict = json.load(open(f'{save_dir}/eval_analysis.json'))
        train_res_dict = eval_dict['train']
        test_res_dict = eval_dict['test']

        train_FN = np.array(train_res_dict['FN'])
        train_FP1 = np.array(train_res_dict['FP1'])
        train_FP2 = np.array(train_res_dict['FP2'])

        train_FN_frames = get_eval_frame_pathes(train_FN, train_data)
        train_FP1_frames = get_eval_frame_pathes(train_FP1, train_data)
        train_FP2_frames = get_eval_frame_pathes(train_FP2, train_data)

        train_FN_res = get_eval_statistic(data_dir, train_FN_frames)
        train_FP1_res = get_eval_statistic(data_dir, train_FP1_frames)
        train_FP2_res = get_eval_statistic(data_dir, train_FP2_frames)

        test_FN = np.array(test_res_dict['FN'])
        test_FP1 = np.array(test_res_dict['FP1'])
        test_FP2 = np.array(test_res_dict['FP2'])

        test_FN_frames = get_eval_frame_pathes(test_FN, test_data)
        test_FP1_frames = get_eval_frame_pathes(test_FP1, test_data)
        test_FP2_frames = get_eval_frame_pathes(test_FP2, test_data)

        test_FN_res = get_eval_statistic(data_dir, test_FN_frames)
        test_FP1_res = get_eval_statistic(data_dir, test_FP1_frames)
        test_FP2_res = get_eval_statistic(data_dir, test_FP2_frames)
        
        statistic_dict = {
            'test_statistic':{'FN':test_FN_res,
                        'FP1': test_FP1_res,
                        'FP2': test_FP2_res},
            'train_statistic':{'FN':train_FN_res,
                        'FP1': train_FP1_res,
                        'FP2': train_FP2_res},
            'test_path': {'FN':test_FN_frames,
                        'FP1': test_FP1_frames,
                        'FP2': test_FP2_frames},
            'train_path': {'FN':train_FN_frames,
                        'FP1': train_FP1_frames,
                        'FP2': train_FP2_frames}
        }

        eval_dict['statistic'] = statistic_dict
        with open(f'{save_dir}/eval_analysis.json', 'w') as f:
            json.dump(eval_dict, f, indent=2)

        plot_eval_statistic(test_FN_res, test_FP1_res, test_FP2_res, 'test', f'{save_dir}/error_analysis_test', figsize=(12, 5))
        plot_eval_statistic(train_FN_res, train_FP1_res, train_FP2_res, 'train', f'{save_dir}/error_analysis_train', figsize=(40, 5))