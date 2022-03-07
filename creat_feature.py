from typing import List
import numpy as np###########
import torch


root_path="/home/sfs/test/py36/OpenPCDet/"
feature_path=root_path+"tools/conf/kitti_data/feature.pth"####################3

feature=torch.load(feature_path)
print("the len of feature is")
print(len(feature),'\n')
print("the dimension of feature is")
print(len(feature[0]))