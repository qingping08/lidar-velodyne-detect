import cv2
import numpy as np
import math

import torch
voxel_1=np.array([3.7029e+00,  1.5913e+00,1.4678e+00])
voxel_2=np.array([3.8298e+00,  1.6649e+00,1.4353e+00])
box_1=np.array(
        [[ 2.2135e+01,  5.3410e+00, -1.9296e+00],
         [ 2.2167e+01,  6.9320e+00, -1.9296e+00],
         [ 2.5869e+01,  6.8568e+00, -1.9296e+00],
         [ 2.5837e+01,  5.2659e+00, -1.9296e+00],
         [ 2.2135e+01,  5.3410e+00, -4.6180e-01],
         [ 2.2167e+01,  6.9320e+00, -4.6180e-01],
         [ 2.5869e+01,  6.8568e+00, -4.6180e-01],
         [ 2.5837e+01,  5.2659e+00, -4.6180e-01]
        ]
         
)
box_2=np.array(
[[22.16653255,5.31210738,-1.95550925],
[22.18716934,6.97681125,-1.95398409],
[26.01617479,6.92964079,-1.94758993],
[25.99553799,5.26493691,-1.9491151],
[22.16411988,5.3108223,-0.5202118],
[22.18475667,6.97552617,-0.51868664],
[26.01376212,6.92835571,-0.51229248],
[25.99312532,5.26365183,-0.51381765]]
         
)
""" def calculate_IOU(box_1,box_2,voxel_1,voxel_2):

    rect1_centre_x=(box_1[0][0]+box_1[2][0])/2
    rect1_centre_y=(box_1[0][1]+box_1[2][1])/2

    rect2_centre_x=(box_2[0][0]+box_2[2][0])/2
    rect2_centre_y=(box_2[0][1]+box_2[2][1])/2

    theta1=-np.arctan((box_1[2][1]-box_1[1][1])/(box_1[2][0]-box_1[1][0]))
    theta2=-np.arctan((box_2[2][1]-box_2[1][1])/(box_2[2][0]-box_2[1][0]))

    rect_1=((rect1_centre_x,rect1_centre_y),(voxel_1[0],voxel_1[1]),theta1)
    rect_2=((rect2_centre_x,rect2_centre_y),(voxel_2[0],voxel_2[1]),theta2)
    #print(rect_1)
    #print(rect_2)
    r1 = cv2.rotatedRectangleIntersection(rect_1, rect_2)  # 区分正负角度，逆时针为负，顺时针为正
    order_pts = cv2.convexHull(r1[1], returnPoints=True)
    area_intersecting = cv2.contourArea(order_pts)
    print(area_intersecting)
    delta_h=abs(min(box_1[4][2],box_2[4][2])-max(box_1[0][2],box_2[0][2]))
    V_intersecting=area_intersecting*delta_h

    V_1=voxel_1[0]*voxel_1[1]*voxel_1[2]
    V_2=voxel_2[0]*voxel_2[1]*voxel_2[2]

    IOU=V_intersecting/(V_1+V_2-V_intersecting)
    return IOU """
def rbbox_to_corners(rbbox):
    # generate clockwise corners and rotate it clockwise
    # 顺时针方向返回角点位置
    cx, cy, x_d, y_d, angle = rbbox
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    corners_x = [-x_d / 2, -x_d / 2, x_d / 2, x_d / 2]
    corners_y = [-y_d / 2, y_d / 2, y_d / 2, -y_d / 2]
    corners = [0] * 8
    for i in range(4):
        corners[2 *
                i] = a_cos * corners_x[i] + \
                     a_sin * corners_y[i] + cx
        corners[2 * i +
                1] = -a_sin * corners_x[i] + \
                     a_cos * corners_y[i] + cy
    return corners


def point_in_quadrilateral(pt_x, pt_y, corners):
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]

    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]

    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]

    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1

    return abab >= abap and abap >= 0 and adad >= adap and adap >= 0


def line_segment_intersection(pts1, pts2, i, j):
    # pts1, pts2 为corners
    # i j 分别表示第几个交点，取其和其后一个点构成的线段
    # 返回为 tuple(bool, pts) bool=True pts为交点
    A, B, C, D, ret = [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]
    A[0] = pts1[2 * i]
    A[1] = pts1[2 * i + 1]

    B[0] = pts1[2 * ((i + 1) % 4)]
    B[1] = pts1[2 * ((i + 1) % 4) + 1]

    C[0] = pts2[2 * j]
    C[1] = pts2[2 * j + 1]

    D[0] = pts2[2 * ((j + 1) % 4)]
    D[1] = pts2[2 * ((j + 1) % 4) + 1]
    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]
    CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]
    CA1 = C[1] - A[1]
    # 叉乘判断方向
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        # 判断方向
        if abc != abd:
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            ret[0] = Dx / DH
            ret[1] = Dy / DH
            return True, ret
    return False, ret


def sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    def _cmp(pt, center):
        vx = pt[0] - center[0]
        vy = pt[1] - center[1]
        d = math.sqrt(vx * vx + vy * vy)
        vx /= d
        vy /= d
        if vy < 0:
            vx = -2 - vx
        return vx

    if num_of_inter > 0:
        center = [0, 0]
        for i in range(num_of_inter):
            center[0] += int_pts[i][0]
            center[1] += int_pts[i][1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        int_pts.sort(key=lambda x: _cmp(x, center))


def area(int_pts, num_of_inter):
    def _trangle_area(a, b, c):
        return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) *
                (b[0] - c[0])) / 2.0

    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(
            _trangle_area(int_pts[0], int_pts[i + 1],
                          int_pts[i + 2]))
    return area_val

def calculate_distance(vector_1,vector_2):
    return np.sqrt(np.sum(np.square(vector_1-vector_2)))

def calculate_IOU(box_1,box_2):
    voxel_1_x=calculate_distance(box_1[0],box_1[3])
    voxel_1_y=calculate_distance(box_1[0],box_1[1])
    voxel_1_z=calculate_distance(box_1[0],box_1[4])
    voxel_1=(voxel_1_x,voxel_1_y,voxel_1_z)

    voxel_2_x=calculate_distance(box_2[0],box_2[3])
    voxel_2_y=calculate_distance(box_2[0],box_2[1])
    voxel_2_z=calculate_distance(box_2[0],box_2[4])
    voxel_2=(voxel_2_x,voxel_2_y,voxel_2_z)
    
    rect1_centre_x=(box_1[0][0]+box_1[2][0])/2
    rect1_centre_y=(box_1[0][1]+box_1[2][1])/2

    rect2_centre_x=(box_2[0][0]+box_2[2][0])/2
    rect2_centre_y=(box_2[0][1]+box_2[2][1])/2

    theta1=-np.arctan((box_1[2][1]-box_1[1][1])/(box_1[2][0]-box_1[1][0]))
    theta2=-np.arctan((box_2[2][1]-box_2[1][1])/(box_2[2][0]-box_2[1][0])) 
    rbbox1 = [rect1_centre_x,rect1_centre_y,voxel_1[0],voxel_1[1],theta1]
    rbbox2 = [rect2_centre_x,rect2_centre_y,voxel_2[0],voxel_2[1],theta2]
    corners1 = rbbox_to_corners(rbbox1)
    corners2 = rbbox_to_corners(rbbox2)
    pts, num_pts = [], 0
    for i in range(4):
        point = [corners1[2 * i], corners1[2 * i + 1]]
        if point_in_quadrilateral(point[0], point[1],
                                  corners2):
            num_pts += 1
            pts.append(point)
    for i in range(4):
        point = [corners2[2 * i], corners2[2 * i + 1]]
        if point_in_quadrilateral(point[0], point[1],
                                  corners1):
            num_pts += 1
            pts.append(point)
    for i in range(4):
        for j in range(4):
            ret, point = line_segment_intersection(corners1, corners2, i, j)
            if ret:
                num_pts += 1
                pts.append(point)
    sort_vertex_in_convex_polygon(pts, num_pts)
    polygon_area = area(pts, num_pts)
    #print('area: {}'.format(polygon_area))
    delta_h=abs(min(box_1[4][2],box_2[4][2])-max(box_1[0][2],box_2[0][2]))
    V_intersecting=polygon_area*delta_h

    V_1=voxel_1[0]*voxel_1[1]*voxel_1[2]
    V_2=voxel_2[0]*voxel_2[1]*voxel_2[2]

    IOU=V_intersecting/(V_1+V_2-V_intersecting)
    return IOU
""" box1=torch.tensor(
            [[ 26.1729,  15.1485,  -1.9422],##################################
         [ 28.0152,  15.1052,  -1.9422],
         [ 27.9047,  10.4031,  -1.9422],
         [ 26.0624,  10.4464,  -1.9422],
         [ 26.1729,  15.1485,   0.0556],
         [ 28.0152,  15.1052,   0.0556],
         [ 27.9047,  10.4031,   0.0556],
         [ 26.0624,  10.4464,   0.0556]],
)
box2=torch.tensor(
            [[ 2.6181e+01,  1.5015e+01, -1.6486e+00],#####################
         [ 2.7874e+01,  1.4989e+01, -1.6609e+00],
         [ 2.7804e+01,  1.0310e+01, -1.5327e+00],
         [ 2.6110e+01,  1.0336e+01, -1.5203e+00],
         [ 2.6196e+01,  1.5070e+01,  3.7481e-01],
         [ 2.7889e+01,  1.5044e+01,  3.6246e-01],
         [ 2.7819e+01,  1.0365e+01,  4.9072e-01],
         [ 2.6126e+01,  1.0391e+01,  5.0307e-01]],
)
voxel_1=[4.7034,   1.8429,   1.9978]
voxel_2=[4.6811,   1.6935,   2.0242]
print(calculate_IOU(box_1,box_2,voxel_1,voxel_2))  ## 0.8631482365351171 """