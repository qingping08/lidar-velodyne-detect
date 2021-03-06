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
from visual_utils import visualize_utils as V


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
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg
ref_3d=torch.tensor([
    #158
    [[  1.1417,   5.1776,  -1.8744],
         [  1.0256,   6.6197,  -1.8744],
         [  4.5363,   6.9022,  -1.8744],
         [  4.6524,   5.4601,  -1.8744],
         [  1.1417,   5.1776,  -0.4232],
         [  1.0256,   6.6197,  -0.4232],
         [  4.5363,   6.9022,  -0.4232],
         [  4.6524,   5.4601,  -0.4232]],

        [[ 26.7328,  -8.4836,  -1.2576],
         [ 28.2236,  -8.4580,  -1.2576],
         [ 28.2820, -11.8719,  -1.2576],
         [ 26.7912, -11.8974,  -1.2576],
         [ 26.7328,  -8.4836,   0.1320],
         [ 28.2236,  -8.4580,   0.1320],
         [ 28.2820, -11.8719,   0.1320],
         [ 26.7912, -11.8974,   0.1320]],

        [[ 35.4683,  18.7528,  -1.8146],
         [ 34.4432,  19.8730,  -1.8146],
         [ 36.6961,  21.9346,  -1.8146],
         [ 37.7212,  20.8144,  -1.8146],
         [ 35.4683,  18.7528,  -0.3676],
         [ 34.4432,  19.8730,  -0.3676],
         [ 36.6961,  21.9346,  -0.3676],
         [ 37.7212,  20.8144,  -0.3676]],

        [[  4.3960,  -2.0863,  -1.6744],
         [  4.4795,  -3.6330,  -1.6744],
         [  0.6078,  -3.8419,  -1.6744],
         [  0.5243,  -2.2952,  -1.6744],
         [  4.3960,  -2.0863,  -0.3102],
         [  4.4795,  -3.6330,  -0.3102],
         [  0.6078,  -3.8419,  -0.3102],
         [  0.5243,  -2.2952,  -0.3102]],

        [[ 26.1346,  22.2073,  -2.1823],
         [ 27.7857,  22.1804,  -2.1823],
         [ 27.7272,  18.5941,  -2.1823],
         [ 26.0761,  18.6210,  -2.1823],
         [ 26.1346,  22.2073,  -0.7805],
         [ 27.7857,  22.1804,  -0.7805],
         [ 27.7272,  18.5941,  -0.7805],
         [ 26.0761,  18.6210,  -0.7805]],

        [[ 20.1034, -25.9484,  -0.9247],
         [ 18.4793, -26.0108,  -0.9247],
         [ 18.3208, -21.8864,  -0.9247],
         [ 19.9448, -21.8240,  -0.9247],
         [ 20.1034, -25.9484,   0.5367],
         [ 18.4793, -26.0108,   0.5367],
         [ 18.3208, -21.8864,   0.5367],
         [ 19.9448, -21.8240,   0.5367]],

        [[  4.4948,  13.8017,  -1.9008],
         [  3.0036,  13.7596,  -1.9008],
         [  2.9020,  17.3597,  -1.9008],
         [  4.3932,  17.4018,  -1.9008],
         [  4.4948,  13.8017,  -0.5940],
         [  3.0036,  13.7596,  -0.5940],
         [  2.9020,  17.3597,  -0.5940],
         [  4.3932,  17.4018,  -0.5940]],

        [[ 35.6129,  12.1530,  -1.5601],
         [ 34.3890,  13.2143,  -1.5601],
         [ 36.9793,  16.2014,  -1.5601],
         [ 38.2032,  15.1401,  -1.5601],
         [ 35.6129,  12.1530,  -0.1187],
         [ 34.3890,  13.2143,  -0.1187],
         [ 36.9793,  16.2014,  -0.1187],
         [ 38.2032,  15.1401,  -0.1187]],

        [[ 28.8922, -16.8787,  -1.3173],
         [ 27.3100, -17.1986,  -1.3173],
         [ 26.5010, -13.1979,  -1.3173],
         [ 28.0832, -12.8780,  -1.3173],
         [ 28.8922, -16.8787,   0.0515],
         [ 27.3100, -17.1986,   0.0515],
         [ 26.5010, -13.1979,   0.0515],
         [ 28.0832, -12.8780,   0.0515]],

        [[ 19.3039,  14.5841,  -2.0399],
         [ 17.7957,  14.4699,  -2.0399],
         [ 17.4852,  18.5689,  -2.0399],
         [ 18.9933,  18.6831,  -2.0399],
         [ 19.3039,  14.5841,  -0.6942],
         [ 17.7957,  14.4699,  -0.6942],
         [ 17.4852,  18.5689,  -0.6942],
         [ 18.9933,  18.6831,  -0.6942]],

        [[  2.3640,  13.8535,  -2.0231],
         [  0.8649,  13.7529,  -2.0231],
         [  0.6328,  17.2111,  -2.0231],
         [  2.1319,  17.3117,  -2.0231],
         [  2.3640,  13.8535,  -0.6373],
         [  0.8649,  13.7529,  -0.6373],
         [  0.6328,  17.2111,  -0.6373],
         [  2.1319,  17.3117,  -0.6373]],

        [[  8.8471,  -5.0835,  -1.4160],
         [  8.7321,  -5.6560,  -1.4160],
         [  7.8109,  -5.4709,  -1.4160],
         [  7.9259,  -4.8984,  -1.4160],
         [  8.8471,  -5.0835,   0.4392],
         [  8.7321,  -5.6560,   0.4392],
         [  7.8109,  -5.4709,   0.4392],
         [  7.9259,  -4.8984,   0.4392]],

        [[ 44.5607,  17.8347,  -0.5669],
         [ 46.2024,  18.0280,  -0.5669],
         [ 46.6447,  14.2711,  -0.5669],
         [ 45.0030,  14.0778,  -0.5669],
         [ 44.5607,  17.8347,   1.0476],
         [ 46.2024,  18.0280,   1.0476],
         [ 46.6447,  14.2711,   1.0476],
         [ 45.0030,  14.0778,   1.0476]],

        [[ 26.1729,  15.1485,  -1.9422],
         [ 28.0152,  15.1052,  -1.9422],
         [ 27.9047,  10.4031,  -1.9422],
         [ 26.0624,  10.4464,  -1.9422],
         [ 26.1729,  15.1485,   0.0556],
         [ 28.0152,  15.1052,   0.0556],
         [ 27.9047,  10.4031,   0.0556],
         [ 26.0624,  10.4464,   0.0556]],

        [[ 34.6017,  20.9076,  -1.9137],
         [ 33.5376,  22.1721,  -1.9137],
         [ 36.6688,  24.8068,  -1.9137],
         [ 37.7328,  23.5423,  -1.9137],
         [ 34.6017,  20.9076,  -0.4887],
         [ 33.5376,  22.1721,  -0.4887],
         [ 36.6688,  24.8068,  -0.4887],
         [ 37.7328,  23.5423,  -0.4887]],

        [[  9.2066,  -4.6585,  -1.4627],
         [  8.9345,  -5.1963,  -1.4627],
         [  8.0816,  -4.7649,  -1.4627],
         [  8.3537,  -4.2271,  -1.4627],
         [  9.2066,  -4.6585,   0.4005],
         [  8.9345,  -5.1963,   0.4005],
         [  8.0816,  -4.7649,   0.4005],
         [  8.3537,  -4.2271,   0.4005]],

        [[ 38.1324,   4.7710,  -1.3481],
         [ 37.8824,   4.1455,  -1.3481],
         [ 37.2928,   4.3812,  -1.3481],
         [ 37.5428,   5.0067,  -1.3481],
         [ 38.1324,   4.7710,   0.4807],
         [ 37.8824,   4.1455,   0.4807],
         [ 37.2928,   4.3812,   0.4807],
         [ 37.5428,   5.0067,   0.4807]],

        [[ 21.1429,  29.7008,  -1.7314],
         [ 20.6946,  31.1007,  -1.7314],
         [ 24.1881,  32.2194,  -1.7314],
         [ 24.6364,  30.8195,  -1.7314],
         [ 21.1429,  29.7008,  -0.2476],
         [ 20.6946,  31.1007,  -0.2476],
         [ 24.1881,  32.2194,  -0.2476],
         [ 24.6364,  30.8195,  -0.2476]],

        [[ 11.3195,  27.8817,  -2.1443],
         [ 11.2767,  26.2836,  -2.1443],
         [  7.4881,  26.3851,  -2.1443],
         [  7.5309,  27.9832,  -2.1443],
         [ 11.3195,  27.8817,  -0.7195],
         [ 11.2767,  26.2836,  -0.7195],
         [  7.4881,  26.3851,  -0.7195],
         [  7.5309,  27.9832,  -0.7195]],

        [[ -1.5455,  -2.4649,  -1.7144],
         [ -2.1236,  -0.9458,  -1.7144],
         [  1.3625,   0.3807,  -1.7144],
         [  1.9406,  -1.1384,  -1.7144],
         [ -1.5455,  -2.4649,  -0.2180],
         [ -2.1236,  -0.9458,  -0.2180],
         [  1.3625,   0.3807,  -0.2180],
         [  1.9406,  -1.1384,  -0.2180]],

        [[ 20.1528, -15.1973,  -1.2271],
         [ 18.5557, -15.4110,  -1.2271],
         [ 18.1062, -12.0521,  -1.2271],
         [ 19.7033, -11.8384,  -1.2271],
         [ 20.1528, -15.1973,   0.2835],
         [ 18.5557, -15.4110,   0.2835],
         [ 18.1062, -12.0521,   0.2835],
         [ 19.7033, -11.8384,   0.2835]],

        [[ 44.6742,  23.0092,  -1.1516],
         [ 44.6289,  24.5724,  -1.1516],
         [ 48.3370,  24.6800,  -1.1516],
         [ 48.3823,  23.1168,  -1.1516],
         [ 44.6742,  23.0092,   0.3483],
         [ 44.6289,  24.5724,   0.3483],
         [ 48.3370,  24.6800,   0.3483],
         [ 48.3823,  23.1168,   0.3483]],
#211
        [[ 2.5992e+01,  2.2292e+01, -2.0130e+00],
         [ 2.7615e+01,  2.2297e+01, -2.0062e+00],
         [ 2.7625e+01,  1.8448e+01, -1.9080e+00],
         [ 2.6002e+01,  1.8443e+01, -1.9148e+00],
         [ 2.5986e+01,  2.2328e+01, -6.2545e-01],
         [ 2.7609e+01,  2.2332e+01, -6.1870e-01],
         [ 2.7619e+01,  1.8483e+01, -5.2050e-01],
         [ 2.5996e+01,  1.8479e+01, -5.2724e-01]],

        [[ 3.5541e+01,  1.1808e+01, -1.3742e+00],
         [ 3.4319e+01,  1.3010e+01, -1.4100e+00],
         [ 3.7353e+01,  1.6092e+01, -1.4758e+00],
         [ 3.8575e+01,  1.4889e+01, -1.4400e+00],
         [ 3.5535e+01,  1.1842e+01, -9.6643e-03],
         [ 3.4313e+01,  1.3045e+01, -4.5505e-02],
         [ 3.7347e+01,  1.6127e+01, -1.1126e-01],
         [ 3.8569e+01,  1.4924e+01, -7.5416e-02]],

        [[ 2.6746e+01, -8.7747e+00, -1.0953e+00],
         [ 2.8305e+01, -8.7390e+00, -1.0896e+00],
         [ 2.8384e+01, -1.2199e+01, -1.0010e+00],
         [ 2.6825e+01, -1.2235e+01, -1.0067e+00],
         [ 2.6740e+01, -8.7389e+00,  3.0625e-01],
         [ 2.8300e+01, -8.7032e+00,  3.1194e-01],
         [ 2.8378e+01, -1.2163e+01,  4.0050e-01],
         [ 2.6819e+01, -1.2199e+01,  3.9482e-01]],

        [[ 2.6046e+01,  1.4760e+01, -1.8362e+00],
         [ 2.7883e+01,  1.4740e+01, -1.8279e+00],
         [ 2.7833e+01,  1.0219e+01, -1.7129e+00],
         [ 2.5997e+01,  1.0239e+01, -1.7211e+00],
         [ 2.6037e+01,  1.4812e+01,  2.3460e-01],
         [ 2.7874e+01,  1.4793e+01,  2.4287e-01],
         [ 2.7825e+01,  1.0272e+01,  3.5795e-01],
         [ 2.5988e+01,  1.0292e+01,  3.4967e-01]],

        [[ 3.2904e+01,  3.4923e+01, -2.1420e+00],
         [ 3.3925e+01,  3.6180e+01, -2.1697e+00],
         [ 3.6832e+01,  3.3820e+01, -2.0972e+00],
         [ 3.5812e+01,  3.2564e+01, -2.0695e+00],
         [ 3.2898e+01,  3.4963e+01, -5.9377e-01],
         [ 3.3918e+01,  3.6219e+01, -6.2150e-01],
         [ 3.6825e+01,  3.3860e+01, -5.4904e-01],
         [ 3.5805e+01,  3.2603e+01, -5.2131e-01]],

        [[ 1.9429e+01,  1.4326e+01, -1.9285e+00],
         [ 1.7742e+01,  1.4237e+01, -1.9333e+00],
         [ 1.7515e+01,  1.8553e+01, -2.0443e+00],
         [ 1.9202e+01,  1.8642e+01, -2.0395e+00],
         [ 1.9423e+01,  1.4362e+01, -5.0223e-01],
         [ 1.7736e+01,  1.4273e+01, -5.0710e-01],
         [ 1.7509e+01,  1.8589e+01, -6.1812e-01],
         [ 1.9196e+01,  1.8678e+01, -6.1325e-01]],

        [[ 2.5808e+01,  3.7675e+01, -2.5201e+00],
         [ 2.7513e+01,  3.7682e+01, -2.5131e+00],
         [ 2.7529e+01,  3.3695e+01, -2.4113e+00],
         [ 2.5824e+01,  3.3688e+01, -2.4184e+00],
         [ 2.5801e+01,  3.7713e+01, -1.0411e+00],
         [ 2.7507e+01,  3.7720e+01, -1.0341e+00],
         [ 2.7523e+01,  3.3732e+01, -9.3234e-01],
         [ 2.5817e+01,  3.3725e+01, -9.3937e-01]],

        [[ 3.4690e+01,  1.4678e+01, -1.4175e+00],
         [ 3.3703e+01,  1.5880e+01, -1.4523e+00],
         [ 3.6405e+01,  1.8097e+01, -1.4974e+00],
         [ 3.7392e+01,  1.6895e+01, -1.4626e+00],
         [ 3.4684e+01,  1.4713e+01, -4.8943e-02],
         [ 3.3697e+01,  1.5915e+01, -8.3772e-02],
         [ 3.6399e+01,  1.8132e+01, -1.2890e-01],
         [ 3.7386e+01,  1.6930e+01, -9.4066e-02]],

        [[ 2.6443e+01, -1.3436e+01, -9.7544e-01],
         [ 2.8119e+01, -1.3271e+01, -9.7255e-01],
         [ 2.8547e+01, -1.7639e+01, -8.5935e-01],
         [ 2.6871e+01, -1.7803e+01, -8.6225e-01],
         [ 2.6436e+01, -1.3397e+01,  5.2816e-01],
         [ 2.8112e+01, -1.3233e+01,  5.3105e-01],
         [ 2.8541e+01, -1.7601e+01,  6.4424e-01],
         [ 2.6864e+01, -1.7765e+01,  6.4135e-01]],

        [[ 1.0842e+01,  2.7595e+01, -2.1395e+00],
         [ 1.0810e+01,  2.5926e+01, -2.0971e+00],
         [ 6.7226e+00,  2.6003e+01, -2.1163e+00],
         [ 6.7541e+00,  2.7672e+01, -2.1587e+00],
         [ 1.0836e+01,  2.7632e+01, -6.8989e-01],
         [ 1.0804e+01,  2.5963e+01, -6.4745e-01],
         [ 6.7164e+00,  2.6040e+01, -6.6669e-01],
         [ 6.7480e+00,  2.7709e+01, -7.0913e-01]],

        [[ 2.6006e+01,  4.3483e+01, -2.6086e+00],
         [ 2.7653e+01,  4.3437e+01, -2.6005e+00],
         [ 2.7537e+01,  3.9331e+01, -2.4963e+00],
         [ 2.5890e+01,  3.9378e+01, -2.5044e+00],
         [ 2.5999e+01,  4.3521e+01, -1.1095e+00],
         [ 2.7646e+01,  4.3475e+01, -1.1014e+00],
         [ 2.7531e+01,  3.9370e+01, -9.9720e-01],
         [ 2.5884e+01,  3.9416e+01, -1.0053e+00]],

        [[ 1.7203e+01,  2.4181e+01, -2.2279e+00],
         [ 1.8746e+01,  2.4300e+01, -2.2244e+00],
         [ 1.9034e+01,  2.0550e+01, -2.1276e+00],
         [ 1.7492e+01,  2.0431e+01, -2.1310e+00],
         [ 1.7197e+01,  2.4219e+01, -7.3931e-01],
         [ 1.8739e+01,  2.4338e+01, -7.3581e-01],
         [ 1.9028e+01,  2.0588e+01, -6.3895e-01],
         [ 1.7486e+01,  2.0469e+01, -6.4245e-01]],

        [[ 3.2492e+01,  4.1457e+01, -2.4709e+00],
         [ 3.3533e+01,  4.2674e+01, -2.4975e+00],
         [ 3.6365e+01,  4.0253e+01, -2.4238e+00],
         [ 3.5324e+01,  3.9036e+01, -2.3972e+00],
         [ 3.2486e+01,  4.1494e+01, -1.0047e+00],
         [ 3.3527e+01,  4.2711e+01, -1.0313e+00],
         [ 3.6359e+01,  4.0290e+01, -9.5757e-01],
         [ 3.5318e+01,  3.9073e+01, -9.3095e-01]],

        [[ 3.7190e+01,  4.2378e+00, -1.1441e+00],
         [ 3.7435e+01,  4.7836e+00, -1.1570e+00],
         [ 3.8072e+01,  4.4981e+00, -1.1470e+00],
         [ 3.7827e+01,  3.9524e+00, -1.1342e+00],
         [ 3.7183e+01,  4.2824e+00,  6.0159e-01],
         [ 3.7428e+01,  4.8281e+00,  5.8871e-01],
         [ 3.8064e+01,  4.5426e+00,  5.9868e-01],
         [ 3.7819e+01,  3.9969e+00,  6.1156e-01]],

        [[ 3.7834e+01,  3.7389e+01, -2.1867e+00],
         [ 3.7248e+01,  3.5851e+01, -2.1500e+00],
         [ 3.3510e+01,  3.7273e+01, -2.2020e+00],
         [ 3.4095e+01,  3.8811e+01, -2.2388e+00],
         [ 3.7828e+01,  3.7426e+01, -7.3447e-01],
         [ 3.7242e+01,  3.5888e+01, -6.9774e-01],
         [ 3.3504e+01,  3.7310e+01, -7.4981e-01],
         [ 3.4089e+01,  3.8848e+01, -7.8654e-01]],

        [[ 4.3653e+01,  4.3313e+00, -1.1080e+00],
         [ 4.3755e+01,  4.9961e+00, -1.1245e+00],
         [ 4.4228e+01,  4.9237e+00, -1.1207e+00],
         [ 4.4126e+01,  4.2588e+00, -1.1041e+00],
         [ 4.3646e+01,  4.3755e+00,  6.2454e-01],
         [ 4.3748e+01,  5.0403e+00,  6.0802e-01],
         [ 4.4220e+01,  4.9678e+00,  6.1187e-01],
         [ 4.4118e+01,  4.3030e+00,  6.2839e-01]],

        [[ 3.5890e+01,  5.2482e+00, -1.2126e+00],
         [ 3.6172e+01,  5.7131e+00, -1.2233e+00],
         [ 3.6785e+01,  5.3411e+00, -1.2112e+00],
         [ 3.6502e+01,  4.8761e+00, -1.2005e+00],
         [ 3.5882e+01,  5.2931e+00,  5.5071e-01],
         [ 3.6165e+01,  5.7581e+00,  5.4004e-01],
         [ 3.6777e+01,  5.3861e+00,  5.5212e-01],
         [ 3.6494e+01,  4.9211e+00,  5.6278e-01]],

        [[ 9.5944e+00,  1.3646e+01, -1.5290e+00],
         [ 9.1817e+00,  1.5136e+01, -1.5687e+00],
         [ 1.2707e+01,  1.6112e+01, -1.5787e+00],
         [ 1.3120e+01,  1.4622e+01, -1.5390e+00],
         [ 9.5879e+00,  1.3685e+01,  6.8554e-04],
         [ 9.1752e+00,  1.5175e+01, -3.9057e-02],
         [ 1.2701e+01,  1.6151e+01, -4.9044e-02],
         [ 1.3113e+01,  1.4661e+01, -9.3008e-03]],

        [[ 2.9907e+01,  1.6867e+01, -1.8782e+00],
         [ 2.8564e+01,  1.7781e+01, -1.9072e+00],
         [ 3.0672e+01,  2.0875e+01, -1.9772e+00],
         [ 3.2015e+01,  1.9961e+01, -1.9482e+00],
         [ 2.9901e+01,  1.6904e+01, -4.3340e-01],
         [ 2.8558e+01,  1.7817e+01, -4.6238e-01],
         [ 3.0666e+01,  2.0912e+01, -5.3238e-01],
         [ 3.2009e+01,  1.9998e+01, -5.0339e-01]]


        ])
ref_corners3d=ref_3d.numpy()

labels=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,##158
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
ref_labels=labels.numpy()

scores=torch.tensor([
        0.9992, 0.9948, 0.9808, 0.9735, 0.9507, 0.9185, 0.9182, 0.8807, 0.8642,
        0.8617, 0.8553, 0.8406, 0.7874, 0.7245, 0.6912, 0.6684, 0.3689, 0.3567,
        0.2520, 0.2009, 0.1902, 0.1202,#158


        0.9990, 0.9971, 0.9969, 0.9967, 0.9961, 0.9852, 0.9716, 0.9551, 0.9410,
        0.9242, 0.8748, 0.8113, 0.7894, 0.4059, 0.2836, 0.2643, 0.2169, 0.1336,
        0.1072])
ref_scores=scores.numpy()

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            #print(data_dict)
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            ##
            #print(pred_dicts)
            ##print(pred_dicts[0]['pred_boxes'])
            points=data_dict['points'][:, 1:]
            if not isinstance(points, np.ndarray):
                points = points.cpu().numpy()
            fig =V. visualize_pts(points)
            fig = V.draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))
            for k in range(ref_labels.min(), ref_labels.max() + 1):
                cur_color = tuple(V.box_colormap[k % len(V.box_colormap)])
                mask = (ref_labels == k)
                fig = V.draw_corners3d(ref_corners3d[mask], fig=fig, color=cur_color, cls=ref_scores[mask], max_num=100)
            mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
            mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
