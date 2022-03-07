import numpy as np
##  0233.bin->0158.bin
##          B       ->     A
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
""" def matrix_Euler_angle(yaw,pitch,roll):
    axis_z=[0,0,1]
    z_rot_matrix=np.linalg.expm(np.cross(np.eye(3), axis_z / np.linalg.norm(axis_z) * yaw))
    axis_y=[0,1,0]
    z_rot_matrix=np.linalg.expm(np.cross(np.eye(3), axis_y / np.linalg.norm(axis_y) * pitch)) """
def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def matrix_Euler_angle(roll,pitch,yaw):
    Rx = rotx(roll)
    Ry = roty(pitch)
    Rz = rotz(yaw)
    R = Rz.dot(Ry.dot(Rx))
    return R
""" def T_lat_lon(lat,lon,alt):
    scale = np.cos(lat * np.pi / 180.)
    er = 6378137#the radius of Earth
    tx = scale * lon * np.pi * er / 180.
    ty = scale * er * \
        np.log(np.tan((90. + lat) * np.pi / 360.))
    tz = alt
    t = np.array([tx, ty, tz])
    return t """
def T_shift(lat1,lon1,alt1,lat2,lon2,alt2):
    er = 6378137#the radius of Earth
    delta_x=er*np.cos(lat1*np.pi/180)*(lon1-lon2)*np.pi/180
    delta_y=er*(lat1-lat2)*np.pi/180
    delta_z=alt1-alt2
    return np.array([-delta_x,-delta_y,-delta_z])
def T_world_2_1(lat1,lon1,alt1,lat2,lon2,alt2,roll2,pitch2,yaw2):
    R=matrix_Euler_angle(roll2,pitch2,yaw2)
    t=T_shift(lat1,lon1,alt1,lat2,lon2,alt2)
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))
def T_world_imu(roll,pitch,yaw):
    R_imu_world=matrix_Euler_angle(roll,pitch,yaw)
    R_world_imu=np.linalg.inv(R_imu_world)
    t=np.array([0,0,0])
    R = R_world_imu.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))
""" x_velo2=1
y_velo2=1
z_velo2=1
#coordinates in vehicle 2
r_velo2=np.array([[x_velo2],[y_velo2],[z_velo2],[1]])
T_imu_velo=np.array([[9.999976e-01,7.553071e-04,-2.035826e-03,-8.086759e-01],\
                    [-7.854027e-04,9.998898e-01,-1.482298e-02,3.195559e-01],\
                    [2.024406e-03,1.482454e-02,9.998881e-01,-7.997231e-01],\
                    [0,0,0,1]])
T_velo_imu=np.linalg.inv(T_imu_velo)

#coordinates in imu2
r_imu2=np.dot(T_velo_imu,r_velo2)
#the T of world 2 to world 1
T_2_to_1=T_world_2_1(49.009347760599,8.4371344308375,114.45652008057,\
                     49.009344871709,8.4371499178574,114.45387268066,\
                     0.073139,-0.000124,-0.2696086732051)
r_world1=np.dot(T_2_to_1,r_imu2)

r_imu1=np.dot(T_world_imu(0.072242,0.001566,-0.2697266732051),r_world1)

r_velo1=np.dot(T_imu_velo,r_imu1) """
def transform_2_1(x_velo2,y_velo2,z_velo2):
    r_velo2=np.array([[x_velo2],[y_velo2],[z_velo2],[1]])
    T_imu_velo=np.array([[9.999976e-01,7.553071e-04,-2.035826e-03,-8.086759e-01],\
                    [-7.854027e-04,9.998898e-01,-1.482298e-02,3.195559e-01],\
                    [2.024406e-03,1.482454e-02,9.998881e-01,-7.997231e-01],\
                    [0,0,0,1]])##B_imu_velo
    T_velo_imu=np.linalg.inv(T_imu_velo)
    r_imu2=np.dot(T_velo_imu,r_velo2)
    """     T_2_to_1=T_world_2_1(49.009347760599,8.4371344308375,114.45652008057,\
                     49.009344871709,8.4371499178574,114.45387268066,\
                     0.073139,-0.000124,-0.2696086732051) """
    T_2_to_1=T_world_2_1(lat1,lon1,alt1,lat2,lon2,alt2,roll2,pitch2,yaw2)
    r_world1=np.dot(T_2_to_1,r_imu2)
    #r_imu1=np.dot(T_world_imu(0.072242,0.001566,-0.2697266732051),r_world1)
    r_imu1=np.dot(T_world_imu(roll1,pitch1,yaw1),r_world1)
    r_velo1=np.dot(T_imu_velo,r_imu1)
    return r_velo1
    