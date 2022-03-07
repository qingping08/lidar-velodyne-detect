import numpy as np

feature_root="/home/sfs/test/py36/OpenPCDet/tools/conf/kitti_data/SSD_feature/env4/env4/"
model_name="resnet"
weather_name="night"
layer_name="f4"

feature_path=feature_root+model_name+'_'+weather_name+'_'+layer_name+'.npy'

feature=np.load(feature_path)

print(feature.shape) 
print(feature.shape[1]) 