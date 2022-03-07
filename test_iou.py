import os
filename1='/home/sfs/kitti_data/2011_09_26/2011_09_26_drive_0009_sync/oxts/data/0000000158.txt'
with open(filename1,'r+',encoding='utf-8') as f1:
    data1 = f1.read().split(' ')  # 读取文件
filename2='/home/sfs/kitti_data/2011_09_26/2011_09_26_drive_0009_sync/oxts/data/0000000233.txt'
with open(filename2,'r+',encoding='utf-8') as f2:
    data2 = f2.read().split(' ')  # 读取文件
lat1=float(data1[0])
lon1=float(data1[1])
alt1=float(data1[2])
roll1=float(data1[3])
pitch1=float(data1[4])
yaw1=float(data1[5])
lat2=float(data2[0])
lon2=float(data2[1])
alt2=float(data2[2])
roll2=float(data2[3])
pitch2=float(data2[4])
yaw2=float(data2[5])
print(lat1+lon1)