import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            #points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pv_rcnn.yaml ',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default="pv_rcnn_8369.pth", help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

print("--------------------------begin calculating---------------------------------")
root_path="/home/sfs/test/py36/OpenPCDet/"
train_txt_path=root_path+"data/kitti/ImageSets/train.txt"
save_path=root_path+"tools/conf/kitti_data/normal_dt_data/"
num=5 
with open(train_txt_path,'r') as f:
    lines=f.readlines()
velodyne_ids=[x.strip() for x in lines]
velodyne_ids=velodyne_ids[0:num]
for i,velodyne_id in enumerate(velodyne_ids):
    args,cfg=parse_config()
    logger = common_utils.create_logger()
    #logger.info('-----------------Create dt-------------------------')
    velodyne_path=root_path+"data/kitti/training/velodyne/"+str(velodyne_id)+".bin"
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(velodyne_path), ext=args.ext, logger=logger
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            data_dict = demo_dataset.collate_batch([data_dict])
            #print(data_dict)
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            save_file_path=save_path+"velodyne_"+str(velodyne_id)+".npy"
            np.save(save_file_path,pred_dicts)
    print(i,"succeed")

""" velodyne_paths=root_path+"data/kitti/training/velodyne/"+str(velodyne_ids)+".bin"
print(velodyne_paths)
print(velodyne_paths[0])
velodyne_paths=[]
for i,velodyne_id in enumerate(velodyne_ids):
    velodyne_paths[i]=root_path+"data/kitti/training/velodyne/"+str(velodyne_id)+".bin"
print(velodyne_paths) """


