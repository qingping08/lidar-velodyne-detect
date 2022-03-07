import numpy as np
import torch
import sys
sys.path.append(r'/home/sfs/test/py36/OpenPCDet/tools')
import IOU
#sys.path.append(r'/home/sfs/test/py36/OpenPCDet/tools/visual_utils')
#import visual_utils.visualize_utils

class_name=1
def choose_car(pred_boxes,pred_scores,pred_labels,class_name):
    boxes=[]
    scores=[]
    labels=[]
    for i,label in enumerate(pred_labels):
        if label==class_name:#car
            scores.append(pred_scores[i])
            labels.append(pred_labels[i])
            boxes.append(pred_boxes[i])
    return boxes,scores,labels

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]
    return corners3d.numpy() if is_numpy else corners3d



root_path="/home/sfs/test/py36/OpenPCDet/"
train_txt_path=root_path+"data/kitti/ImageSets/train.txt"
det_path=root_path+"tools/conf/kitti_data/normal_dt_data/"
gt_path=root_path+"tools/conf/kitti_data/normal_data/"

test_file_id="000003"
print("-----------------------------------------"+test_file_id+"------------------------------------------")
gt_file_path=gt_path+test_file_id+".npy"
dt_file_path=det_path+"velodyne_"+test_file_id+".npy"

gt_data=np.load(gt_file_path,allow_pickle=True).item()
dt_data=np.load(dt_file_path,allow_pickle=True)

gt_boxes=gt_data['annos']['gt_boxes_lidar']################# choose  car
#pred_dicts=dt_data[0]['pred_boxes'].cpu().numpy()
#print(torch.from_numpy(gt_boxes))
pred_scores=dt_data[0]['pred_scores'].cpu().numpy()
pred_labels=dt_data[0]['pred_labels'].cpu().numpy()
pred_boxes=dt_data[0]['pred_boxes']#.cpu().numpy()

car_ids=np.argwhere(pred_labels==class_name)

#print(type(dt_boxes)) 
#print(len(dt_boxes))
#print(gt_boxes)
#print(gt_data['image'])
# len(R)就是当前类别的gt目标个数，det表示是否检测到，初始化为false。
ref_corners3d_dts=boxes_to_corners_3d(pred_boxes)
ref_corners3d_gts=boxes_to_corners_3d(gt_boxes)

dt_boxes,scores,labels=choose_car(ref_corners3d_dts,pred_scores,pred_labels,class_name)

npos=len(dt_boxes)
det = [False] * len(gt_boxes)
ovthresh=0.5

tp=0
fp=0

num=0
for i,dt_box in enumerate(dt_boxes):
    overlaps=[]
    for  j,ref_corners3d_gt in enumerate(ref_corners3d_gts):
        overlap=IOU.calculate_IOU(dt_box.cpu().numpy(),ref_corners3d_gt)
        overlaps.append(overlap)
    #print(overlaps)
    print('the det_car '+str(i)+' iou is ')
    print(overlaps)
    ovmax=np.max(overlaps)
    #print(ovmax)
    jmax=np.argmax(overlaps)
    #print(jmax)
    if ovmax>ovthresh:
        if not det[jmax]:
            tp+=1
    else:
        fp+=1 

print('tp =',tp)
print('fp =',fp)
p_rec=tp/(tp+fp)
print(p_rec)    

        