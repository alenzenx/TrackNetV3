import os
import glob
import cv2
import csv
import shutil
original_raw_data = 'raw_data2'
target_folder = 'TrackNetV2_Dataset/test'



def video2img(video, csvv, output_frame_path2, output_csv_path, output_video_path, match, video_name):
    # video to img
    with open(csvv, 'r') as file:
        lines = file.readlines()[1:]

        csv_content = []
        for line in lines:
            frame, vis, x, y = line.strip().split(',')
            csv_content.append((int(frame), int(vis), float(x), float(y)))

    count = 0
    num_data = len(csv_content)
    cap = cv2.VideoCapture(video)
    success, image = cap.read()
    while success:
        if count >= num_data:
            break
        cv2.imwrite(os.sep.join([output_frame_path2, '%d.png' %(count)]), image)
        success, image = cap.read()
        count += 1
    
    # copy csv and convert it
    csv_file_path = output_csv_path + video_name +'_ball.csv'
    # os.system('pause')
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame','Visibility','X','Y'])
        for i in range(0,num_data,1):
            aaa = list(csv_content[i])
            if(aaa[1]==0):
                aaa[2] = 0
                aaa[3] = 0
            aaa[2] = aaa[2]*1024*1.25
            aaa[3] = aaa[3]*576*1.25
            writer.writerow(aaa)
        
    # copy video and rename
    video_file_path = output_video_path + video_name +'.mp4'
    shutil.copyfile(video, video_file_path)





videos = sorted(glob.glob(os.path.join(original_raw_data, '*.mp4')))
csvs = sorted(glob.glob(os.path.join(original_raw_data, '*.csv')))
if(len(videos) < 2):
    print("The number of videos is less than 2, please increase the number of videos")
    print("影片數小於2，請增加影片數。\n\n")
    os._exit()
# print(videos)
# print(csvs)
match = 1
print("==========Convert Start==========")
for video, csvv in zip(videos, csvs):
    v_name = os.path.split(video)[-1]
    csv_name = os.path.split(csvv)[-1]
    if v_name[:-4] != csv_name[:-4]:
        raise NameError("Video files and csv files are not corresponded")
    print("Convert Video: {}".format(video))
    output_path = '/match%d'%(match)
    output_path = target_folder + output_path
    output_csv_path = output_path + '/csv'
    output_frame_path = output_path + '/frame'
    output_video_path = output_path + '/video'

    this_videofile_name = '/%d_01_00'%(match)

    output_frame_path2 = output_frame_path + this_videofile_name
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
        os.mkdir(output_csv_path)
        os.mkdir(output_frame_path)
        os.mkdir(output_video_path)
        os.mkdir(output_frame_path2)
    
    video2img(video, csvv, output_frame_path2, output_csv_path, output_video_path, match, this_videofile_name)

    match += 1
    print("==========Convert End==========")