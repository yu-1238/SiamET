
import os
import pandas as pd
import glob, cv2, torch
import numpy as np
from os.path import realpath, dirname, join
import cv2
# from trackers.tracker_GRU import SiamRPNTracker
from trackers.tracker_siamET import SiamETTracker
from trackers.utils import get_axis_aligned_bbox, cxy_wh_2_rect


model_path='../models/OTB100/model.pth'

et_path1= '../models/OTB100/ET_model.pth.tar'
et_path2= '../models/OTB100/ET_model.pth.tar'
et_path0= '../models/OTB100/ET_model.pth.tar'

tracker= SiamETTracker(model_path, et_path1, 0, 4, et_path2, et_path0)
# image and init box

image_files = sorted(glob.glob('../Basketball/img/*.jpg'))
gt_bboxes = pd.read_csv(os.path.join('../Basketball/', "groundtruth_rect.txt"), sep='\t|,| ',
                        header=None, names=['xmin', 'ymin', 'width', 'height'],
                        engine='python')
# init_rbox = [334.02,128.36,438.19,188.78,396.39,260.83,292.23,200.41]
init_rbox=gt_bboxes.iloc[0].values
#init_rbox = [334,128,438,188,396,260,292,200]
# init_rbox=np.array(init_rbox)

im = cv2.imread(image_files[0])  # HxWxC
init_rect =init_rbox
# init_rect = cv2.selectROI('updatenete', im, False, False)
[cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)

# tracker init
target_pos, target_sz = np.array([cx, cy]), np.array([w, h])


# tracking and visualization
toc = 0
first=True
for f, image_file in enumerate(image_files):
    if first:
        im = cv2.imread(image_file)
        state = tracker.init(im, init_rbox)
        first=False
    else:
        im = cv2.imread(image_file)
        tic = cv2.getTickCount()
        state = tracker.track(im)  # track
        toc += cv2.getTickCount() - tic
        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        res = [int(l) for l in res]
        cv2.rectangle(im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 0, 255), 3)
        cv2.imshow('SiamET', im)
        cv2.waitKey(1)

print('Tracking Speed {:.1f}fps'.format((len(image_files)-1)/(toc/cv2.getTickFrequency())))

