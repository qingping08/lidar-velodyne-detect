import numpy as np
import  matplotlib.pyplot as plt
from get_recall2 import get_test_recall_list
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def sigmoid(x):
    return 1/(1+np.exp(-x))
model_name='resnet'
cachefile='../environmental sensitivity/annotations_cache_MVI_40213/annots.pkl'
weathers = ['normal', 'rain', 'light', 'fog', 'speed']
weathers_ch=['原始数据','雨天数据','强光数据','加雾数据','高速数据']
dense_root='./conf_res_fig/'
rec_root='./conf_res_fig2/'
dense=np.load(dense_root + '/' + 'test_conf_list_' + model_name + '.npy')
rec=np.load(rec_root + '/' + 'test_conf_list_' + model_name + '.npy')
overthersh=0.5
imgset_name='../environmental sensitivity/imageset_MVI_40213.txt'
with open(imgset_name,'r') as f:
    img_names=f.readlines()
    img_names=[x.strip() for x in img_names]

prob_array=np.zeros((len(weathers),len(img_names)))
class_prob_array=np.zeros((len(weathers),len(img_names)))
recall_array=np.zeros((len(weathers),len(img_names)))
for wea_idx,weather in enumerate(weathers):
    det_file='../environmental sensitivity/result_MVI_40213/resnet/'+weather+'/car.txt'
    recall_array[wea_idx]=np.array(get_test_recall_list(cachefile,det_file,img_names,overthersh))
    img_rec_prob = {}
    img_rec_num = {}
    with open(det_file, 'r') as f:
        lines = f.readlines()
        conf_list = []
        N = len(lines)
        for line in lines:
            info = line.strip().split(' ')
            prob = float(info[1])
            img_name = info[0]
            if img_name in img_rec_prob:
                img_rec_prob[img_name] += prob
                img_rec_num[img_name] += 1
            else:
                img_rec_prob[img_name] = prob
                img_rec_num[img_name] = 1
        for img_idx, img_name in enumerate(img_names):
            if img_name in img_rec_prob:
                prob = img_rec_prob[img_name]
                n = img_rec_num[img_name]
                prob_array[wea_idx][img_idx] = prob / n * np.log(n + 1)
                class_prob_array[wea_idx][img_idx] = prob / n
            else:
                prob_array[wea_idx][img_idx] = 0
                class_prob_array[wea_idx][img_idx] = 0

plt.figure()
conf=np.sqrt(sigmoid(7*rec*dense)*prob_array)

for i in [0,3]:
    plt.plot(conf[i],label=weathers_ch[i])
# plt.xlabel('frame id')
# plt.ylabel('conf')
# plt.title('confidence in different environment')
plt.xlabel('帧编号')
plt.ylabel('置信度')
plt.title('不同环境下语义信息的置信度差异')
plt.legend()

# for i in [0,3]:
#     plt.figure()
#     plt.plot(conf[i],label='conf')
#     plt.plot(recall_array[i],label='recall')
#     plt.plot(class_prob_array[i], label='class prob')
#     plt.legend()
#     plt.xlabel('t')
#     plt.ylabel('value')
#     plt.title('confidence and recall in %s environment' % weathers[i])
#     plt.legend()