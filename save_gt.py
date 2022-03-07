import pickle
import numpy as np
#Uninstalling numpy-1.19.2: 1.16.2
root_path="kitti_data/normal_data/"
data_path=root_path+"test.npy"

data=np.load(data_path,allow_pickle=True)

for i in range(0,len(data)):
    data_cloud=data[i]
    image_id=data[i]['image']['image_idx']
    save_path=root_path+str(image_id)+".npy"
    np.save(save_path,data_cloud)





