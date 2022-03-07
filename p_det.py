import numpy as np
import torch

def choose_car(pred_scores,pred_labels):
    scores=[]
    labels=[]
    for i,label in enumerate(pred_labels):
        if label==1:#car
            scores.append(pred_scores[i])
            labels.append(pred_labels[i])
    return scores,labels
num=5
root_path="/home/sfs/test/py36/OpenPCDet/"
train_txt_path=root_path+"data/kitti/ImageSets/train.txt"
det_path=root_path+"tools/conf/kitti_data/normal_dt_data/"

with open(train_txt_path,'r') as f:
    lines=f.readlines()

velodyne_ids=[x.strip() for x in lines]
velodyne_ids=velodyne_ids[0:num]
print(velodyne_ids)
img_rec_prob = {}
img_rec_num = {}
prob_array={}
class_prob_array={}

for i,velodyne_id in enumerate(velodyne_ids):
    det_file=det_path+"velodyne_"+str(velodyne_id)+".npy"
    pred_dicts=np.load(det_file,allow_pickle=True)
    #print(pred_dicts)
    pred_scores=pred_dicts[0]['pred_scores']
    pred_labels=pred_dicts[0]['pred_labels']
    pred_boxes=pred_dicts[0]['pred_boxes']
    #print(pred_scores)
    #print(pred_labels)
    #print(pred_boxes)
    scores,labels=choose_car(pred_scores,pred_labels)
    #print(len(labels))
    img_rec_prob[velodyne_id]=float(np.sum(scores))
    img_rec_num[velodyne_id]=len(labels)

    prob = img_rec_prob[velodyne_id]
    n = img_rec_num[velodyne_id]
    #print(prob)
    #print(n) 
    prob_array[velodyne_id] = prob / n * np.log(n + 1)
    class_prob_array[velodyne_id] = prob / n
print(prob_array)


