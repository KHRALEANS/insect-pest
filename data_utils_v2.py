import numpy as np
import pandas as pd
import typing
from PIL import Image
import os, shutil
import json


def make_coco_dic(dataFrame: pd.DataFrame, imgNames: np.ndarray, imgIndex: typing.Dict, datasetDir: str) -> typing.Tuple[typing.Dict, pd.DataFrame]:
    
    labels = {'images':[],'annotations':[],'categories':[]}

    image_indices = [(dataFrame['文件名'][i] in imgNames) for i in range(0,len(dataFrame['文件名']))]
    df_part = dataFrame[image_indices]

    for i, file_name in enumerate(imgNames):
        id = imgIndex[file_name]
        img_file = os.path.join(datasetDir, file_name)
        img = Image.open(img_file)
        img_shape = img.size
        width = img_shape[0]
        height = img_shape[1]
        tmp_images = {
            "id": id,
            "file_name": file_name,
            "height": height,
            "width": width
        }
        labels['images'].append(tmp_images)       

    category_names = dataFrame['虫子名称'].unique()
    insect_index = {}
    for i in range(0,len(category_names)):
        id = i + 1
        name = category_names[i]
        insect_index[name] = id

    for key in insect_index.keys():
        id = insect_index[key]
        name = key
        if id == 1:
            supercategory = 'insect'
        else:
            supercategory = 'pest'
        tmp_categories = {
            "id": id,
            "name": name,
            "supercategory": supercategory
        }
        labels['categories'].append(tmp_categories)

    for i in range(0,df_part.shape[0]):
        id = i
        image_id = imgIndex[df_part.iloc[i]['文件名']]
        category_id = insect_index[df_part.iloc[i]['虫子名称']]
        if np.isnan(df_part.iloc[i]['中心点x坐标']):
            # bbox = []
            # area = 0
            continue
        else:
            x = int(df_part.iloc[i]['左上角x坐标'])
            y = int(df_part.iloc[i]['左上角y坐标'])
            w = int(df_part.iloc[i]['右下角x坐标']) - int(df_part.iloc[i]['左上角x坐标'])
            h = int(df_part.iloc[i]['右下角y坐标']) - int(df_part.iloc[i]['左上角y坐标'])
            bbox = [x, y, w, h]
            area = w * h
        tmp_anno = {
            "id": id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        }
        labels['annotations'].append(tmp_anno)
              
    return [labels, df_part]


def save_to_json(labels: typing.Dict, save_file_name: str):
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)


    with open(save_file_name, 'w') as lf:
        json.dump(labels, lf, cls=NpEncoder)

    return


def copy_images(imageNames: typing.List, oriDir: str, destDir: str):

    for fname in imageNames:
        src = os.path.join(oriDir, fname)
        dst = os.path.join(destDir, fname)
        shutil.copyfile(src, dst)

    return