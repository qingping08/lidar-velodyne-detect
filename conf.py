from get_recall2 import get_recall_list, get_conf, get_test_recall_list
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import matplotlib.pyplot as plt
import numpy as np

model_name_dict = {'mobilenet_v1': 'ssd_mobilenet_v1_coco_11_06_2017', \
                   'mobilenet_v2': 'ssd_mobilenet_v2_coco_2018_03_29', \
                   'inception': 'ssd_inception_v2_coco_2018_01_28', \
                   'resnet': 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03', \
                   'vgg': ' '}
model_name_list = list(model_name_dict.keys())
# es_path = '../environmental sensitivity/es/DETRAC_MVI_40213/es1612007926.xls'
# det_root = 'E:/lyr/code/confidence/result/coco/'
# train_feature_root = 'E:/lyr/code/confidence/feature/coco/'
# test_feature_root = 'E:/lyr/code/environmental sensitivity/feature/DETRAC_MVI_40213/'
# cachefile = 'E:/lyr/code/confidence/annotations_cache_coco/annots.pkl'
# # imagesetfile = 'imageset.txt'
# ovthresh = 0.5
weathers = ['normal', 'rain', 'light', 'fog', 'speed']
res_root = './conf_res_fig2/'
# if not os.path.exists(res_root):
#     os.makedirs(res_root)
# with open(imagesetfile, 'r') as f:
#     lines = f.readlines()
#     # 待检测图像文件名字存于数组imagenames,长度1000。
# imagenames = [x.strip() for x in lines]
#
# es_dic = {}
# with open(es_path, 'r') as f:
#     lines = f.readlines()
#     for line in lines[1:]:
#         info = line.strip().split('\t')
#         model_name = info[0]
#         es_list = info[1:-1]
#         es_list = [float(x) for x in es_list]
#         es_dic[model_name] = es_list

res_path_pattern=res_root+'test_conf_list_{}.npy'
net_weather_conf=[]
for model_name in model_name_list:
    res_path=res_path_pattern.format(model_name)
    net_weather_conf.append(np.load(res_path))

for net_conf, model_name in zip(net_weather_conf,model_name_list):
    train_path=res_root+'/'+'train_conf_'+model_name+'.npy'
    train_conf=np.load(train_path)
    plt.figure()
    plt.plot(train_conf,label='train')
    for i,weather in enumerate(weathers):
        plt.plot(net_conf[i],label=weather)
    plt.title('train and different environment data confidence of %s' % model_name)
    plt.legend()

#
# for i,weather in enumerate(weathers):
#     plt.figure()
#     for weather_conf, model_name in zip(net_weather_conf,model_name_list):
#         plt.plot(weather_conf[i],label=model_name)
#     plt.title('confidence of %s weather data in different networks' % weather)
#     plt.legend()

# imagesetfile='../environmental sensitivity/imageset_MVI_40213.txt'
# ovthresh=0.5
# cachefile='../environmental sensitivity/annotations_cache_MVI_40213/annots.pkl'
# det_root='../environmental sensitivity/result_MVI_40213/'
# with open(imagesetfile, 'r') as f:
#     lines = f.readlines()
#     # 待检测图像文件名字存于数组imagenames,长度1000。
# imagenames = [x.strip() for x in lines]
# test_recall_list=[]
# for i,model_name in enumerate(model_name_list):
#     for j,weather in enumerate(weathers):
#         recall_list=get_test_recall_list(cachefile,det_root, model_name,imagenames,ovthresh,weather)
#         conf_list=net_weather_conf[i][j]
#         plt.figure()
#         plt.plot(recall_list,label='recall')
#         plt.plot(conf_list,label='conf')
#         plt.title('recall and confidence of %s in the %s environment' % (model_name, weather))
#         plt.legend()
#         plt.savefig(res_root+'/'+ 'recall_conf_'+model_name+'_'+weather+'.png')
#         plt.close()