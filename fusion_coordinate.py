import mayavi.mlab as mlab
import numpy as np
import torch
IOU=0.7
ref_3d=torch.tensor([

        [[ 2.2135e+01,  5.3410e+00, -1.9296e+00],
         [ 2.2167e+01,  6.9320e+00, -1.9296e+00],
         [ 2.5869e+01,  6.8568e+00, -1.9296e+00],
         [ 2.5837e+01,  5.2659e+00, -1.9296e+00],
         [ 2.2135e+01,  5.3410e+00, -4.6180e-01],
         [ 2.2167e+01,  6.9320e+00, -4.6180e-01],
         [ 2.5869e+01,  6.8568e+00, -4.6180e-01],
         [ 2.5837e+01,  5.2659e+00, -4.6180e-01]],

[[22.16653255,5.31210738,-1.95550925],
[22.18716934,6.97681125,-1.95398409],
[26.01617479,6.92964079,-1.94758993],
[25.99553799,5.26493691,-1.9491151],
[22.16411988,5.3108223,-0.5202118],
[22.18475667,6.97552617,-0.51868664],
[26.01376212,6.92835571,-0.51229248],
[25.99312532,5.26365183,-0.51381765]],

        ])
scores=torch.tensor([0.9942, 0.9981])
fusion_coor=(ref_3d[0]*scores[0]+ref_3d[1]*scores[1])/(scores[0]+scores[1])
fusion_score=scores[0]*IOU+scores[1]*(1-IOU)
print(fusion_coor)
print(fusion_score)
def fusion(scores,box_1,box_2,IOU):
        fusion_coor=(ref_3d[0]*scores[0]+ref_3d[1]*scores[1])/(scores[0]+scores[1])
        fusion_score=scores[0]*IOU+scores[1]*(1-IOU) 
        return fusion_coor,fusion_score
        