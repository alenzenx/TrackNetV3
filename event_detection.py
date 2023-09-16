import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.optimize import curve_fit
import csv
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.signal import find_peaks
import argparse

parser = argparse.ArgumentParser(description = 'Auto rally cut with TrackNet predict result')
parser.add_argument('--input_video', type = str, help = 'input video name')
parser.add_argument('--input_csv', type = str, help = 'input csv name')
args = parser.parse_args()

filename = args.input_csv
list1=[]
frames=[]
realx=[]
realy=[]
points=[]

def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle

def get_point_line_distance(point, line):
    point_x = point[0]
    point_y = point[1]
    line_s_x = line[0]
    line_s_y = line[1]
    line_e_x = line[2]
    line_e_y = line[3]
    if line_e_x - line_s_x == 0:
        return math.fabs(point_x - line_s_x)
    if line_e_y - line_s_y == 0:
        return math.fabs(point_y - line_s_y)
    #斜率
    k = (line_e_y - line_s_y) / (line_e_x - line_s_x)
    #截距
    b = line_s_y - k * line_s_x
    #带入公式得到距离dis
    dis = math.fabs(k * point_x - point_y + b) / math.pow(k * k + 1, 0.5)
    return dis

with open(filename, newline='') as csvFile:
    rows = csv.reader(csvFile, delimiter=',')
    num = 0
    count=0
    for row in rows:
        list1.append(row)
    front_zeros=np.zeros(len(list1))
    for i in range(1,len(list1)):
        frames.append(int(float(list1[i][0])))
        realx.append(int(float(list1[i][2])))
        realy.append(int(float(list1[i][3])))
        if int(float(list1[i][2])) != 0:
            front_zeros[num] = count
            points.append((int(float(list1[i][2])),int(float(list1[i][3])),int(float(list1[i][0]))))
            num += 1
        else:
            count += 1

# 羽球2D軌跡點
points = np.array(points)
x, y, z = points.T

Predict_hit_points = np.zeros(len(frames))
ang = np.zeros(len(frames))
# from scipy.signal import find_peaks
peaks, properties = find_peaks(y, prominence=5)

print(peaks)

if(len(peaks) >= 5):
    lower = np.argmin(y[peaks[0]:peaks[1]])
    if (y[peaks[0]] - lower) < 5:
        peaks = np.delete(peaks,0)

    lower = np.argmin(y[peaks[-2]:peaks[-1]])
    if (y[peaks[-1]] - lower) < 5:
        peaks = np.delete(peaks,-1)

print()
print('Serve : ')
start_point = 0

for i in range(len(y)-1):
    if((y[i] - y[i+1]) / (z[i+1] - z[i]) >= 5):
        start_point = i+front_zeros[i]
        Predict_hit_points[int(start_point)] = 1
        print(int(start_point))
        break

print('End : ')
end_point = 10000

print('Predict points : ')
plt.plot(z,y*-1,'-')
for i in range(len(peaks)):
    print(peaks[i]+int(front_zeros[peaks[i]]),end=',')
    if(peaks[i]+int(front_zeros[peaks[i]]) >= start_point and peaks[i]+int(front_zeros[peaks[i]]) <= end_point):
        Predict_hit_points[peaks[i]+int(front_zeros[peaks[i]])] = 1


#打擊的特定frame = peaks[i]+int(front_zeros[peaks[i]])
print()
print('Extra points : ')
for i in range(len(peaks)-1):
    start = peaks[i]
    end = peaks[i+1]+1
    upper=[]
    plt.plot(z[start:end],y[start:end]*-1,'-')
    lower = np.argmin(y[start:end]) #找到最低谷(也就是從最高點開始下墜到下一個擊球點),以此判斷扣殺或平球軌跡
    for j in range(start+lower, end+1):
        if(j-(start+lower) > 5) and (end - j > 5):
            if (y[j] - y[j-1])*3 < (y[j+1] - y[j]):
                print(j, end=',')
                ang[j+int(front_zeros[j])] = 1

            point = [x[j],y[j]]
            line=[x[j-1],y[j-1],x[j+1],y[j+1]]
            # if get_point_line_distance(point,line) > 2.5:
            if angle([x[j-1],y[j-1], x[j],y[j]],[x[j],y[j], x[j+1],y[j+1]]) > 130:
                print(j, end=',')
                ang[j+int(front_zeros[j])] = 1

ang, _ = find_peaks(ang, distance=15)
#final_predict, _  = find_peaks(Predict_hit_points, distance=10)
for i in ang:
    Predict_hit_points[i] = 1
Predict_hit_points, _ = find_peaks(Predict_hit_points, distance=5)
final_predict = []
for i in (Predict_hit_points):
    final_predict.append(i)

print()
print('Final predict : ')
print(list(final_predict))

with open(filename[:-4] + '_event.csv','w', newline='') as csvfile1:
    h = csv.writer(csvfile1)
    h.writerow(['frame','event'])
    for i in range(len(frames)):
        if i in final_predict:
            h.writerow([frames[i],1])
        else:
            h.writerow([frames[i],0])

plt.show()





