import os
import json
import time
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from dataset import Badminton_Dataset
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='TrackNetV2')
parser.add_argument('--num_frame', type=int, default=3)
parser.add_argument('--input_type', type=str, default='2d', choices=['2d', '3d'])
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--tolerance', type=float, default=4)
parser.add_argument('--save_dir', type=str, default='exp')
parser.add_argument('--resume_training', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
args = parser.parse_args()
param_dict = vars(args)

torch.backends.cudnn.benchmark = True #cudnn找尋最佳卷積算法

model_name = args.model_name
num_frame = args.num_frame
input_type = args.input_type
epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
tolerance = args.tolerance
save_dir = args.save_dir
resume_training = args.resume_training
debug = args.debug
save_dir = f'{save_dir}_debug' if debug else save_dir
display_step = 4 if debug else 100

if resume_training:
    checkpoint = torch.load(f'{save_dir}/model_cur.pt')
    param_dict = checkpoint['param_dict']
    model_name = param_dict['model_name']
    num_frame = param_dict['num_frame']
    input_type = param_dict['input_type']
    # learning_rate = param_dict['learning_rate']
    tolerance = param_dict['tolerance']
    save_dir = param_dict['save_dir']
    debug = param_dict['debug']
    save_dir = f'{save_dir}_debug' if debug else save_dir

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load dataset
print(f'Data dir: {data_dir}')
print(f'Data input type: {input_type}')
train_dataset = Badminton_Dataset(root_dir=data_dir, split='train', mode=input_type, num_frame=num_frame, slideing_step=1, debug=debug)
eval_test_dataset = Badminton_Dataset(root_dir=data_dir, split='test', mode=input_type, num_frame=num_frame, slideing_step=num_frame, debug=debug)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size, drop_last=True, pin_memory=False) #已更改pin_memory=False
eval_loader = DataLoader(eval_test_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size, drop_last=False, pin_memory=False) #已更改pin_memory=False
if __name__ == '__main__':

    # create model
    model = get_model(model_name, num_frame, input_type).cuda()
    model_summary(model, model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if not resume_training:
        loss_list = []
        test_acc_dict = {'TP':[], 'TN': [], 'FP1': [], 'FP2': [], 'FN': [], 'accuracy': [], 'precision': [], 'recall': []}
        start_epoch = 0
        max_test_acc = 0.
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss_list = checkpoint['loss_list']
        test_acc_dict = checkpoint['test_acc']
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = np.max(test_acc_dict['accuracy'])
        print(f'Resume training from epoch {start_epoch}.')

    # training loop
    train_start_time = time.time()
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        loss = train(epoch, model, optimizer, WeightedBinaryCrossEntropy, train_loader, input_type, display_step, save_dir)
        loss_list.append(loss)
        torch.save(dict(epoch=epoch,
                        model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        param_dict=param_dict,
                        loss_list=loss_list,
                        test_acc=test_acc_dict), f'{save_dir}/model_cur.pt')

        accuracy, precision, recall, TP, TN, FP1, FP2, FN = evaluation(model, eval_loader, tolerance, input_type)
        TP, TN, FP1, FP2, FN = len(TP), len(TN), len(FP1), len(FP2), len(FN)
        #print(f'\nacc: {accuracy:.4f}\tprecision: {precision:.4f}\trecall: {recall:.4f}\tTP: {TP}\tTN: {TN}\tFP1: {FP1}\tFP2: {FP2}\tFN: {FN}')
        
        test_acc_dict['TP'].append(TP)
        test_acc_dict['TN'].append(TN)
        test_acc_dict['FP1'].append(FP1)
        test_acc_dict['FP2'].append(FP2)
        test_acc_dict['FN'].append(FN)
        test_acc_dict['accuracy'].append(accuracy)
        test_acc_dict['precision'].append(precision)
        test_acc_dict['recall'].append(recall)

        print(f'[epoch: {epoch})]\tEpoch runtime: {(time.time() - start_time) / 3600.:.2f} hrs')
        plot_result(loss_list, None, test_acc_dict, num_frame, save_dir, model_name)
        
        if test_acc_dict['accuracy'][-1] >= max_test_acc:
            max_test_acc = test_acc_dict['accuracy'][-1]
            torch.save(dict(epoch=epoch,
                            model_state_dict=model.state_dict(),
                            optimizer_state_dict=optimizer.state_dict(),
                            param_dict=param_dict,
                            loss_list=loss_list,
                            test_acc=test_acc_dict), f'{save_dir}/model_best.pt')

    torch.save(dict(epoch=epoch,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    param_dict=param_dict,
                    loss_list=loss_list,
                    test_acc=test_acc_dict), f'{save_dir}/model_last.pt')

    print(f'runtime: {(time.time() - train_start_time) / 3600.:.2f} hrs')
    print('Done......')