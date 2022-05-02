# %%
# modi
image2test = '/root/autodl-tmp/测试数据/测试图像文件'

import numpy as np
import pandas as pd

from mmdet.apis import init_detector, inference_detector
import mmcv

# modi
config_file = 'mmdetection/configs/yolox/my_yolox_v7_pre70_200e.py'
# modi
checkpoint_file = 'mmdetection/work_dirs/my_yolox_v7_pre70_200e/best_bbox_mAP_epoch_75.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')
# test an image
# modi
img = '/root/autodl-tmp/测试数据/测试图像文件/00012.jpg'
result = inference_detector(model, img)

# %%
import os


image_names = os.listdir(image2test)
# %%
class_bbox_candidates = []
thr = 0.7

for image_name in image_names:
    
    img = os.path.join(image2test, image_name)
    result = inference_detector(model, img)
    for i, bbox_candis in enumerate(result):
        for bbox_candi in bbox_candis:
            if bbox_candi[4] > thr:
                x1 = bbox_candi[0]
                y1 = bbox_candi[1]
                x2 = bbox_candi[2]
                y2 = bbox_candi[3]
                bbox_class = i
                score = bbox_candi[4]

                class_bbox_candidate = {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'bbox_class': bbox_class,
                    'score': score,
                    'image_name': image_name
                }
                class_bbox_candidates.append(class_bbox_candidate)

len(class_bbox_candidates)
# %%
# 原始标注目录
# modi
original_ann_dir = '/root/autodl-tmp/2022.04.06(正式数据)/附件2/'
# 原始标注文件
# modi
original_ann = os.path.join(original_ann_dir, '图片虫子位置详情表.csv')

ori_df = pd.read_csv(original_ann, encoding='gbk')

class_name_list = ['无', '二化螟', '二点委夜蛾', '棉铃虫', '褐飞虱属', '白背飞虱', '八点灰灯蛾', '蝼蛄', '蟋蟀', '甜菜夜蛾', '黄足猎蝽', '稻纵卷叶螟', '甜菜白带野螟', '黄毒蛾', '石蛾', '大黑鳃金龟', '粘虫', '稻螟蛉', '甘蓝夜蛾', '地老虎', '大螟', '瓜绢野螟', '线委夜蛾', '水螟蛾', '紫条尺蛾', '歧角螟', '草地螟', '豆野螟', '干纹冬夜蛾']
name_dic = {v:k for v,k in enumerate(class_name_list)}

continuousIndex2officialIndex = []
for i in range(0,29):
    continuousIndex2officialIndex.append(ori_df[ori_df['虫子名称'] == name_dic[i]]['虫子编号'].unique()[0])

# %%
result_df = pd.DataFrame(columns=['序号','文件名','虫子编号','中心点x坐标','中心点y坐标','左上角x坐标','左上角y坐标','右下角x坐标','右下角y坐标'])

for i, candi in enumerate(class_bbox_candidates):
    medium_coor_x = int((candi['x1']+candi['x2'])/2)
    medium_coor_y = int((candi['y1']+candi['y2'])/2)

    row = {
        '序号' : i,
        '文件名': candi['image_name'],
        '虫子编号': continuousIndex2officialIndex[candi['bbox_class']],
        '中心点x坐标': medium_coor_x,
        '中心点y坐标': medium_coor_y,
        '左上角x坐标': int(candi['x1']),
        '左上角y坐标': int(candi['y1']),
        '右下角x坐标': int(candi['x2']),
        '右下角y坐标': int(candi['y2'])
    }
    result_df = result_df.append(row, ignore_index=True)

# %%
result_df.to_csv('newTest_result2.csv', index=False, encoding='gbk')
result_df

# %%
result3_df = result_df.groupby(['文件名','虫子编号']).size().reset_index(name='虫子数量')
result3_df

# %%
result3_df.to_csv('newTest_result3.csv', index=False, encoding='gbk')
# %%
