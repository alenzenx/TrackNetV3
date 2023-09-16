import os
import cv2
import math
import torch
import parse
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageSequence

HEIGHT = 288
WIDTH = 512
data_dir = 'TrackNetV2_Dataset'

###################################  Helper Functions ###################################
# def list_dirs(directory):
#     """ Extension of os.listdir which return the directory pathes including input directory.

#         args:
#             directory - A str of directory path

#         returns:
#             A list of directory pathes
#     """
#     return sorted([os.path.join(directory, path) for path in os.listdir(directory)])
def list_dirs(directory):
    """Return a sorted list of directory paths including input directory."""
    return sorted([os.path.normpath(os.path.join(directory, path)).replace("\\", "/") for path in os.listdir(directory)])

def get_model(model_name, num_frame, input_type):
    """ Create model by name and the configuration parameter.

        args:
            model_name - A str of model name
            num_frame - An int specifying the length of a single input sequence
            input_type - A str specifying input type
                '2d' for stacking all the frames at RGB channel dimesion result in shape (H, W, F*3)
                '3d' for stacking all the frames at extra dimesion result in shape (F, H, W, 3)

        returns:
            model - A keras.Model
            input_shape - A tuple specifying the input shape (for model.summary)
    """
    # Import model
    if model_name == 'TrackNetV2':
        from model import TrackNetV2 as TrackNet

    if model_name in ['TrackNetV2']:
        model = TrackNet(in_dim=num_frame*3, out_dim=num_frame)
    
    return model

def model_summary(model, model_name):
    total_count = 0
    total_byte_coubt = 0
    for param in model.parameters():
        total_count += param.nelement()
        total_byte_coubt += param.nelement()*param.element_size()
    print('=======================================')
    print(f'Model: {model_name}')
    print(f'Number of parameters: {total_count}.')
    print(f'Memory usage of : {total_byte_coubt/1024/1024:.4f} MB')
    print('=======================================')

def frame_first_RGB(input, input_type):
    """ Helper function for transforming x to cv image format.

        args:
            input - A numpy.ndarray of RGB image sequences with shape (N, input_shape)
            input_type - A str specifying input type
                '2d' for stacking all the frames at RGB channel dimesion result in shape (H, W, F*3)
                '3d' for stacking all the frames at extra dimesion result in shape (F, H, W, 3)

        returns:
            A numpy.ndarray of RGB image sequences with shape (N, F, H, W, 3)
    """
    assert len(input.shape) > 3
    if input_type == '2d': # (N, F*3, H ,W)
        input = np.transpose(input, (0, 2, 3, 1)) # (N, H ,W, F*3)
    else: # (N, 3, F, H ,W)
        return np.transpose(input, (0, 2, 3, 4, 1))
    
    # Case of input_type == '2d'
    num_frame = int(input.shape[-1]/3)
    tmp_img = np.array([]).reshape(0, num_frame, HEIGHT, WIDTH, 3)
    for n in range(input.shape[0]):
        tmp_frame = np.array([]).reshape(0, HEIGHT, WIDTH, 3)
        for f in range(0, input.shape[-1], 3):
            img = input[n, :, :, f:f+3]
            tmp_frame = np.concatenate((tmp_frame, img.reshape(1, HEIGHT, WIDTH, 3)), axis=0)
        tmp_img = np.concatenate((tmp_img, tmp_frame.reshape(1, num_frame, HEIGHT, WIDTH, 3)), axis=0)
    
    return tmp_img

def frame_first_RGBD(input, input_type):
    """ Helper function for transforming x to cv image format.

        args:
            input - A numpy.ndarray of RGBD image sequences with shape (N, input_shape)
            input_type - A str specifying input type
                '2d' for stacking all the frames at RGB channel dimesion result in shape (H, W, F*3)
                '3d' for stacking all the frames at extra dimesion result in shape (F, H, W, 3)

        returns:
            A numpy.ndarray of RGB image sequences with shape (N, F, H, W, 3)
    """
    assert len(input.shape) > 3
    if input_type == '2d': 
        # (N, F*4, H ,W)
        input = np.transpose(input, (0, 2, 3, 1)) # (N, H ,W, F*4)
    else: 
        # (N, 4, F, H ,W)
        input = input[:, :-1, :, :, :]
        return np.transpose(input, (0, 2, 3, 4, 1))
    
    # Case of input_type == '2d'
    num_frame = int(input.shape[-1]/4)
    tmp_img = np.array([]).reshape(0, num_frame, HEIGHT, WIDTH, 3)
    for n in range(input.shape[0]):
        tmp_frame = np.array([]).reshape(0, HEIGHT, WIDTH, 3)
        for f in range(0, input.shape[-1], 4):
            img = input[n, :, :, f:f+3]
            tmp_frame = np.concatenate((tmp_frame, img.reshape(1, HEIGHT, WIDTH, 3)), axis=0)
        tmp_img = np.concatenate((tmp_img, tmp_frame.reshape(1, num_frame, HEIGHT, WIDTH, 3)), axis=0)
    
    return tmp_img

def frame_first_Gray(input, input_type):
    """ Helper function for transforming y to cv image format.

        args:
            input - A numpy.ndarray of gray scale image sequences with shape (N, input_shape)
            input_type - A str specifying input type
                '2d' for stacking all the frames at RGB channel dimesion result in shape (H, W, F*3)
                '3d' for stacking all the frames at extra dimesion result in shape (F, H, W, 3)
        returns:
            img - A numpy.ndarray of scale imag sequences with shape (N, F, H, W)
    """
    assert len(input.shape) > 3
    if input_type == '2d':
        # (N, F, H ,W)
        return input
    else: 
        # (N, 1, F, H ,W)
        return np.squeeze(input, axis=1)

def get_num_frames(video_file):
    """ Return the number of frames in the video.

        args:
            video_file - A str of video file path with format '{data_dir}/{split}/match{match_id}/video/{rally_id}.mp4

        returns:
            A int specifying the number of frames in the video
    """
    # video_file: 
    assert video_file[-4:] == '.mp4'
    print(video_file)
    match_dir, rally_id = parse.parse('{}/video/{}.mp4', video_file) #需要修改
    frame_dir = f'{match_dir}/frame/{rally_id}'
    assert os.path.exists(frame_dir)

    return len(os.listdir(frame_dir))

def generate_frames(video_file):
    """ Sample frames from the video.

        args:
            video_file - A str of video file path with format '{data_dir}/{split}/match{match_id}/video/{rally_id}.mp4
    """
    try:
        assert video_file[-4:] == '.mp4'
        match_dir, rally_id = parse.parse('{}/video/{}.mp4', video_file)
        csv_file = f'{match_dir}/csv/{rally_id}_ball.csv'
        assert os.path.exists(video_file) and os.path.exists(csv_file)
    except:
        print(f'{video_file} no match csv file.')
        return

    frame_dir = f'{match_dir}/frame/{rally_id}'
    if not os.path.exists(frame_dir):
        # Haven't process
        os.makedirs(frame_dir)
    else:
        label_df = pd.read_csv(csv_file, encoding='utf8')
        if len(list_dirs(frame_dir)) != len(label_df):
            # Some error occur
            shutil.rmtree(frame_dir)
            os.makedirs(frame_dir)
        else:
            # Already processed.
            return

    label_df = pd.read_csv(csv_file, encoding='utf8')
    cap = cv2.VideoCapture(video_file)
    num_frames = 0
    success = True

    # Sample frames until video end or exceed the number of labels
    while success and num_frames != len(label_df):
        success, image = cap.read()
        if success:
            cv2.imwrite(f'{frame_dir}/{num_frames}.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            num_frames += 1

def get_eval_frame_pathes(tuple_array, data):
    """ Get frame pathes according to the evaluation tuple results.

        args:
            tuple_array - A numpy.ndarray of the evaluation tuple results
                each tuple specifying (sequence_id, frame_no)
            data - A dictionary which stored the information for building dataset
                data['filename']: A numpy.ndarray of frame pathe sequences with shape (N, F)
                data['coordinates']: A numpy.ndarray of coordinate sequences with shape (N, F, 2)
                data['visibility']: A numpy.ndarray of visibility sequences with shape (N, F) - 

        returns:
            A list of frame pathes
    """
    path_list = []
    for (i, f) in tqdm(tuple_array):
        path_list.append(data['filename'][i][f])
    return sorted(path_list)

def get_eval_statistic(data_dir, path_list):
    """ Count the number of frame pathes from each rally.

        args:
            data_dir - A str of the root directory of the dataset
            path_list - A list of frame pathes

        returns:
            A dictionary specipying the statistic
                each pair specifying {'{match_id}_{rally_id}': path_count}
    """
    res_dict = {}
    format_string = data_dir + '/{}/match{}/frame/{}/{}.png'
    for path in tqdm(path_list):
        _, m_id, c_id, _ = parse.parse(format_string, path)
        key = f'{m_id}_{c_id}'
        if key not in res_dict.keys():
            res_dict[key] = 1
        else:
            res_dict[key] += 1
    res_dict = sorted(res_dict.items(), key=lambda x:x[1], reverse=True)
    return {k: c for k, c in res_dict}

##################################  Training Functions ##################################
def WeightedBinaryCrossEntropy(y, y_pred):
    # epsilon = 1e-7
    loss = (-1)*(torch.square(1 - y_pred) * y * torch.log(torch.clamp(y_pred, 1e-7, 1)) + torch.square(y_pred) * (1 - y) * torch.log(torch.clamp(1 - y_pred, 1e-7, 1)))
    return torch.mean(loss) # (N, 3, 288, 512)

def FocalWBCE(y, y_pred):
    # epsilon = 1e-7
    gamma = 1
    loss = (-1)*(torch.square(1 - y_pred) * (torch.clamp(1 - y_pred, 1e-7, 1)** gamma) * y * torch.log(torch.clamp(y_pred, 1e-7, 1)) + torch.square(y_pred)* ((torch.clamp(y_pred, 1e-7, 1)) ** gamma) * (1 - y) * torch.log(torch.clamp(1 - y_pred, 1e-7, 1)))
    return torch.mean(loss) # (N, 3, 288, 512)

def train(epoch, model, optimizer, loss_fn, data_loader, input_type, display_step, save_dir):
    model.train()
    data_prob = tqdm(data_loader)
    epoch_loss = []
    for step, (i, x, y, c) in enumerate(data_prob):
        x, y = x.float().cuda(), y.float().cuda()
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y, y_pred)
        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if (step + 1) % display_step == 0:
            show_prediction(x, y, y_pred, c, input_type, save_dir)
            data_prob.set_description(f'Epoch [{epoch}]')
            data_prob.set_postfix(loss=loss.item())
    return float(np.mean(epoch_loss))

def evaluation(model, data_loader, tolerance, input_type):
    model.eval()
    data_prob = tqdm(data_loader)
    TP, TN, FP1, FP2, FN = [], [], [], [], []
    for step, (i, x, y, c) in enumerate(data_prob):
        x, y = x.float().cuda(), y.float().cuda()
        with torch.no_grad():
            y_pred = model(x)
        y_pred = y_pred > 0.5
        # y_pred = y_pred > 0.4
        tp, tn, fp1, fp2, fn = get_confusion_matrix(i, y_pred, y, c, tolerance, input_type=input_type)
        TP.extend(tp)
        TN.extend(tn)
        FP1.extend(fp1)
        FP2.extend(fp2)
        FN.extend(fn)
        
        data_prob.set_description(f'Evaluation')
        data_prob.set_postfix(TP=len(TP), TN=len(TN), FP1=len(FP1), FP2=len(FP2), FN=len(FN))
    
    accuracy, precision, recall = get_metric(len(TP), len(TN), len(FP1), len(FP2), len(FN))
    print(f'\nacc: {accuracy:.4f}\tprecision: {precision:.4f}\trecall: {recall:.4f}\tTP: {len(TP)}\tTN: {len(TN)}\tFP1: {len(FP1)}\tFP2: {len(FP2)}\tFN: {len(FN)}')
    return accuracy, precision, recall, TP, TN, FP1, FP2, FN

def get_confusion_matrix(indices, y_pred, y_true, y_coor, tolerance, input_type='3d'):
    """ Helper function Generate input sequences from frames.

        args:
            indices - A tf.EagerTensor of indices for sequences
            y_pred - A tf.EagerTensor of predicted heatmap sequences
            y_true - A tf.EagerTensor of ground-truth heatmap sequences
            y_coor - A tf.EagerTensor of ground-truth coordinate sequences
            tolerance - A int speicfying the tolerance for FP1
            input_type - A str specifying input type
                '2d' for stacking all the frames at RGB channel dimesion result in shape (H, W, F*3)
                '3d' for stacking all the frames at extra dimesion result in shape (F, H, W, 3)
        returns:
            TP, TN, FP1, FP2, FN - Lists of tuples of all the prediction results
                                    each tuple specifying (sequence_id, frame_no)
    """
    TP, TN, FP1, FP2, FN = [], [], [], [], []
    y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy()
    y_pred = frame_first_Gray(y_pred, input_type)
    y_true = frame_first_Gray(y_true, input_type)
    for n in range(y_pred.shape[0]):
        num_frame = y_pred.shape[1]
        for f in range(num_frame):
            y_p = y_pred[n][f]
            y_t = y_true[n][f]
            c_t = y_coor[n][f]
            if np.amax(y_p) == 0 and np.amax(y_t) == 0:
                # True Negative: prediction is no ball, and ground truth is no ball
                TN.append((int(indices[n]), int(f)))
            elif np.amax(y_p) > 0 and np.amax(y_t) == 0:
                # False Positive 2: prediction is ball existing, but ground truth is no ball
                FP2.append((int(indices[n]), int(f)))
            elif np.amax(y_p) == 0 and np.amax(y_t) > 0:
                # False Negative: prediction is no ball, but ground truth is ball existing
                FN.append((int(indices[n]), int(f)))
            elif np.amax(y_p) > 0 and np.amax(y_t) > 0:
                # both prediction and ground truth are ball existing
                h_pred = y_p * 255
                h_true = y_t * 255
                h_pred = h_pred.astype('uint8')
                h_true = h_true.astype('uint8')
                #h_pred
                (cnts, _) = cv2.findContours(h_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for i in range(len(rects)):
                    area = rects[i][2] * rects[i][3]
                    if area > max_area:
                        max_area_idx = i
                        max_area = area
                target = rects[max_area_idx]
                cx_pred, cy_pred = int(target[0] + target[2] / 2), int(target[1] + target[3] / 2)
                cx_true, cy_true = int(c_t[0]), int(c_t[1])
                dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
                if dist > tolerance:
                    # False Positive 1: prediction is ball existing, but is too far from ground truth
                    FP1.append((int(indices[n]), int(f)))
                else:
                    # True Positive
                    TP.append((int(indices[n]), int(f)))
    return TP, TN, FP1, FP2, FN

def get_metric(TP, TN, FP1, FP2, FN):
    """ Helper function Generate input sequences from frames.

        args:
            TP, TN, FP1, FP2, FN - Each float specifying the count for each result type of prediction

        returns:
            accuracy, precision, recall - Each float specifying the value of metric
    """
    try:
        accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
    except:
        accuracy = 0
    try:
        precision = TP / (TP + FP1 + FP2)
    except:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except:
        recall = 0
    return accuracy, precision, recall

##################################  Prediction Functions ##################################
def get_frame_unit(frame_list, num_frame):
    """ Sample frames from the video.

        args:
            frame_list - A str of video file path with format '{data_dir}/{split}/match{match_id}/video/{rally_id}.mp4

        return:
            frames - A tf.Tensor of a mini batch input sequence
    """
    batch = []
    # Get the resize scaler
    h, w, _ = frame_list[0].shape
    h_ratio = h / HEIGHT
    w_ratio = w / WIDTH
    
    def get_unit(frame_list):
        """ Generate an input sequence from frame pathes and labels.

            args:
                frame_list - A numpy.ndarray of single frame sequence with shape (F,)

            returns:
                frames - A numpy.ndarray of resized frames with shape (H, W, 3*F)
        """
        frames = np.array([]).reshape(0, HEIGHT, WIDTH)

        # Process each frame in the sequence
        for img in frame_list:
            img = cv2.resize(img, (WIDTH, HEIGHT))
            img = np.moveaxis(img, -1, 0)
            frames = np.concatenate((frames, img), axis=0)
        
        return frames
    
    # Form a mini batch of input sequence
    for i in range(0, len(frame_list), num_frame):
        frames = get_unit(frame_list[i: i+num_frame])
        frames /= 255.
        batch.append(frames)

    batch = np.array(batch)
    return torch.FloatTensor(batch)

def get_object_center(heatmap):
    """ Get coordinates from the heatmap.

        args:
            heatmap - A numpy.ndarray of a single heatmap with shape (H, W)

        returns:
            ints specifying center coordinates of object
    """
    if np.amax(heatmap) == 0:
        # No respond in heatmap
        return 0, 0
    else:
        # Find all respond area in the heapmap
        (cnts, _) = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in cnts]

        # Find largest area amoung all contours
        max_area_idx = 0
        max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
        for i in range(len(rects)):
            area = rects[i][2] * rects[i][3]
            if area > max_area:
                max_area_idx = i
                max_area = area
        target = rects[max_area_idx]
    
    return int((target[0] + target[2] / 2)), int((target[1] + target[3] / 2))

def get_pred_type(cx_pred, cy_pred, cx, cy, tolerance):
    """ Get the result type of the prediction.

        args:
            cx_pred, cy_pred - ints specifying the predicted coordinates
            cx, cy - ints specifying the ground-truth coordinates
            tolerance - A int speicfying the tolerance for FP1

        returns:
            A str specifying the result type of the prediction
    """
    pred_has_ball = False if (cx_pred == 0 and cy_pred == 0) else True
    gt_has_ball = False if (cx == 0 and cy == 0) else True
    if  not pred_has_ball and not gt_has_ball:
        return 'TN'
    elif pred_has_ball and not gt_has_ball:
        return 'FP2'
    elif not pred_has_ball and gt_has_ball:
        return 'FN'
    else:
        dist = math.sqrt(pow(cx_pred-cx, 2)+pow(cy_pred-cy, 2))
        if dist > tolerance:
            return 'FP1'
        else:
            return 'TP'

################################  Visualization Functions ################################

def plot_result(loss_list=None, train_acc_dict=None, test_acc_dict=None, num_frame=3, save_dir='', model_name=''):
    """ Plot training performance.

        args:
            loss_list - A list of epoch losses
            train_acc_dict - A dictionary which stored statistic of evaluation on training set
                structure {'TP':[], 'TN': [], 'FP1': [], 'FP2': [], 'FN': [], 'accuracy': [], 'precision': [], 'recall': []}
            test_acc_dict - A dictionary which stored statistic of evaluation on testing set
                structure {'TP':[], 'TN': [], 'FP1': [], 'FP2': [], 'FN': [], 'accuracy': [], 'precision': [], 'recall': []}
            num_frame - An int specifying the length of a single input sequence
            save_dir - A str specifying the save directory
            model_name - A str of model name
    """
    # Plot training epoch losses
    if loss_list:
        plt.title(f'{model_name} (f = {num_frame})\nTraining Loss (WBCE)')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(loss_list)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/loss.jpg')
        plt.clf()

    # Plot accuracy, precision, recall result from evaluation
    plt.title(f'{model_name} (f = {num_frame})\nPerformance')
    if test_acc_dict:
        # test_acc, test_precision, test_recall = np.max(test_acc_dict['accuracy']), np.max(test_acc_dict['precision']), np.max(test_acc_dict['recall'])
        test_acc = np.max(test_acc_dict['accuracy']) #新增的
        index_of_test = np.where(test_acc_dict['accuracy'] == test_acc)[0][0]
        test_precision = test_acc_dict['precision'][index_of_test]
        test_recall = test_acc_dict['recall'][index_of_test]


        plt.plot(test_acc_dict['accuracy'], label='test_accuracy')
        plt.plot(test_acc_dict['precision'], label='test_precision')
        plt.plot(test_acc_dict['recall'], label='test_recall')
    if train_acc_dict:
        # train_acc, train_precision, train_recall = np.max(train_acc_dict['accuracy']), np.max(train_acc_dict['precision']), np.max(train_acc_dict['recall'])
        train_acc = np.max(train_acc_dict['accuracy'])
        index_of_train = np.where(train_acc_dict['accuracy'] == train_acc)[0][0]
        train_precision = train_acc_dict['precision'][index_of_train]
        train_recall = train_acc_dict['recall'][index_of_train]
        
        plt.plot(train_acc_dict['accuracy'], label='train_accuracy')
        plt.plot(train_acc_dict['precision'], label='train_precision')
        plt.plot(train_acc_dict['recall'], label='train_recall')
        
    if train_acc_dict and test_acc_dict:
        plt.xlabel(f'epoch\ntrain  accuracy: {train_acc*100.:.2f} %  precision: {train_precision*100.:.2f} %  recall: {train_recall*100.:.2f} %\n test  accuracy: {test_acc*100.:.2f} %  precision: {test_precision*100.:.2f} %  recall: {test_recall*100.:.2f} %')
    elif test_acc_dict:
        plt.xlabel(f'epochn\n test  accuracy: {test_acc*100.:.2f} %  precision: {test_precision*100.:.2f} %  recall: {test_recall*100.:.2f} %')
    elif train_acc_dict:
        plt.xlabel(f'epochn\n test  accuracy: {train_acc*100.:.2f} %  precision: {train_precision*100.:.2f} %  recall: {train_recall*100.:.2f} %')
    else:
        pass
    plt.ylabel('metric')
    plt.ylim((0.,1.))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/performance.jpg')
    plt.close()

def plot_eval_statistic(FN_res, FP1_res, FP2_res, split, save_file, figsize=(12, 5)):
    """ Plot the distribution of FN, FP1,and FP2 in all rallies.

        args:
            FN_res, FP1_res, FP2_res - Dictionaries which stored the statistic of each prediction result type
                each pair specifying {'{match_id}_{rally_id}': path_count}
            split - A str specify the split of dataset
            save_file - A str specifying the save file name
            figsize - A tuple specifying the size of figure with shape (W, H)
    """
    rally_key = sorted(FN_res.keys())
    FN_list, FP1_list, FP2_list = [], [], []
    # Ensure every rally has value
    for k in rally_key:
        if k in FN_res.keys():
            FN_list.append(FN_res[k])
        else:
            FN_list.append(0)
        if k in FP1_res.keys():
            FP1_list.append(FP1_res[k])
        else:
            FP1_list.append(0)
        if k in FP2_res.keys():
            FP2_list.append(FP2_res[k])
        else:
            FP2_list.append(0)
    
    # Plot stack bar chart
    width = 0.8
    x_tick = np.arange(len(rally_key))
    FN_list, FP1_list, FP2_list = np.array(FN_list), np.array(FP1_list), np.array(FP2_list)
    total_count = FN_list+FP1_list+FP2_list
    plt.figure(figsize=figsize)
    plt.title(f'{split} Set Error Analysis')
    plt.xlabel('clip label')
    plt.ylabel('frame count')
    plt.ylim((0.,np.max(total_count)+60))
    plt.bar(x_tick, FN_list, color='b', label='FN', width=width)
    plt.bar(x_tick, FP1_list, bottom=FN_list, color='g', label='FP1', width=width)
    plt.bar(x_tick, FP2_list, bottom=FN_list+FP1_list, color='r', label='FP2', width=width)
    plt.xticks(x_tick, rally_key, rotation=90)
    for i, c in zip(x_tick, total_count):
        plt.text(x=i-width , y=c+10 , s=c, fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_file}.png')
    plt.close()

def show_prediction(x, y, y_pred, y_coor, input_type, save_dir):
    """ Visualize the inupt sequence with its predicted heatmap.
        Save as a gif image.

        args:
            x - A tf.EagerTensor of input sequences
            y - A tf.EagerTensor of ground-truth heatmap sequences
            y_pred - A tf.EagerTensor of predicted heatmap sequences
            y_coor - A tf.EagerTensor of ground-truth coordinate sequences
            input_type - A str specifying input type
                '2d' for stacking all the frames at RGB channel dimesion result in shape (H, W, F*3)
                '3d' for stacking all the frames at extra dimesion result in shape (F, H, W, 3)
            save_dir - A str specifying the save directory
    """
    imgs = []
    x, y, y_pred, y_coor = x.detach().cpu().numpy(), y.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), y_coor.detach().cpu().numpy()

    # Transform to cv image format (N, F, H , W, C)
    x = frame_first_RGB(x, input_type)
    y = frame_first_Gray(y, input_type)
    y_pred = frame_first_Gray(y_pred, input_type)

    # Only plot the first sequence in the mini-batch
    x, y, y_pred, y_coor = x[0], y[0], y_pred[0], y_coor[0]
    y_map = y_pred > 0.5

    # Scale value from [0, 1] to [0, 255]
    x = x * 255
    y = y * 255
    y_p = y_pred * 255
    y_m = y_map * 255
    x = x.astype('uint8')
    y = y.astype('uint8')
    y_p = y_p.astype('uint8')
    y_m = y_m.astype('uint8')
    
    # Write image sequence to gif
    for f in range(y_coor.shape[0]):
        # Stack channels to form RGB images
        tmp_y = cv2.cvtColor(y[f], cv2.COLOR_GRAY2BGR)
        tmp_pred = cv2.cvtColor(y_p[f], cv2.COLOR_GRAY2BGR)
        tmp_map = cv2.cvtColor(y_m[f], cv2.COLOR_GRAY2BGR)
        tmp_x = x[f]
        assert tmp_x.shape == tmp_y.shape == tmp_pred.shape == tmp_map.shape

        # Mark ground-truth label
        if int(y_coor[f][0]) > 0 and int(y_coor[f][1]) > 0:
            cv2.circle(tmp_x, (int(y_coor[f][0]), int(y_coor[f][1])), 2, (255, 0, 0), -1)
        up_img = cv2.hconcat([tmp_x, tmp_y])
        down_img = cv2.hconcat([tmp_pred, tmp_map])
        img = cv2.vconcat([up_img, down_img])

        # Cast cv image to PIL image for saving gif format
        img = Image.fromarray(img)
        imgs.append(img)
        imgs[0].save(f'{save_dir}/pred_cur.gif', format='GIF', save_all=True, append_images=imgs[1:], duration=1000, loop=0)

