# %%
import json


train_ann_file = 'mmdetection/data/pest-coco/annotations_v2/train_labels.json'
data_root_dir = '/root/autodl-tmp/datasets/pest-coco/AUG'
f = open(train_ann_file)
train_ann_dic = json.load(f)
train_ann_dic['categories']
# %%
import os


for insect_category in train_ann_dic['categories']:
    pest_dir = os.path.join(data_root_dir,str(insect_category['id']))
    isExist = os.path.exists(pest_dir)
    if not isExist:
        os.mkdir(pest_dir)
        print(f'path {pest_dir} created!')
    else:
        print(f'path {pest_dir} exists')

    
# %%
from PIL import Image


im = Image.open('mmdetection/data/pest-coco/train_v2/00006.jpg')
print(im.format, im.size, im.mode)

# %%
for key in train_ann_dic:
    print(key)
# %%
train_ann_dic['annotations'][0]
# %%
train_ann_dic['annotations'][0]['bbox']
# %%
train_ann_dic['images'][1]


# %%
images_index = [-1] * 100
for i, image_ele in enumerate(train_ann_dic['images']):
    while image_ele['id'] > len(images_index) - 1:
        images_index += [-1] * 100
    images_index[image_ele['id']] = i

# %%
img_dir = 'mmdetection/data/pest-coco/train_v2/'

img_name = train_ann_dic['images'][images_index[train_ann_dic['annotations'][0]['image_id']]]['file_name']

img_file = os.path.join(img_dir, img_name)

img_file
# %%
import numpy as np
im = Image.open(img_file)

pix = np.array(im)
# %%
bbox = train_ann_dic['annotations'][0]['bbox']

x1, x2, y1, y2 = bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3]

# %%
pest_slice = pix[y1:y2,x1:x2,:]
# %%
pest_im = Image.fromarray(pest_slice)

# %%
'''# ranned onece
# crop train image
for i, annotation_ele in enumerate(train_ann_dic['annotations']):
    # get original train image
    img_name = train_ann_dic['images'][images_index[annotation_ele['image_id']]]['file_name']
    img_file = os.path.join(img_dir, img_name)

    # get image pixel
    im = Image.open(img_file)
    pix = np.array(im)

    # crop insect pixel
    bbox = annotation_ele['bbox']
    x1, x2, y1, y2 = bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3]
    pest_slice = pix[y1:y2,x1:x2,:]
    pest_im = Image.fromarray(pest_slice)

    # dir name denote category_id ，file name denote annotaion index
    cropped_img_name = str(i)+'.jpg'
    outfile = os.path.join(data_root_dir, str(annotation_ele['category_id']), cropped_img_name)
    pest_im.save(outfile)
'''
# %%
train_ann_dic['categories'][4]
# %%
# original val dataset annotations
val_ann_file = 'mmdetection/data/pest-coco/annotations_v2/val_labels.json'
f = open(val_ann_file)
val_ann_dic = json.load(f)

# original val image dir
val_img_dir = 'mmdetection/data/pest-coco/val_v2/'

val_images_index = [-1] * 100
for i, image_ele in enumerate(val_ann_dic['images']):
    while image_ele['id'] > len(val_images_index) - 1:
        val_images_index += [-1] * 100
    val_images_index[image_ele['id']] = i

# %%
'''# ran once
for i, annotation_ele in enumerate(val_ann_dic['annotations']):
    # get original train image
    img_name = val_ann_dic['images'][val_images_index[annotation_ele['image_id']]]['file_name']
    img_file = os.path.join(val_img_dir, img_name)

    # get image pixel
    im = Image.open(img_file)
    pix = np.array(im)

    # crop insect pixel
    bbox = annotation_ele['bbox']
    x1, x2, y1, y2 = bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3]
    pest_slice = pix[y1:y2,x1:x2,:]
    pest_im = Image.fromarray(pest_slice)

    # dir name denote category_id ，file name denote annotaion index
    cropped_img_name = str(i)+ '_val' +'.jpg'
    outfile = os.path.join(data_root_dir, str(annotation_ele['category_id']), cropped_img_name)
    pest_im.save(outfile)
'''
# %%
val_ann_dic['images']
# %%
val_ann_dic['annotations'][0]
# %%
# count pest num for every image
numPest_per_imageId = [0] * len(images_index)

for i,ele in enumerate(images_index):
    if ele < 0:
        numPest_per_imageId[i] = ele

# %%
for i, annotation_ele in enumerate(train_ann_dic['annotations']):
    numPest_per_imageId[annotation_ele['image_id']] += 1


# %%
numImg_pestExist = 0
numPest = 0
avePest_perImg_pestExist = 0

for i,num in enumerate(numPest_per_imageId):
    if num < 0:
        continue
    if num > 0:
        numImg_pestExist += 1
        numPest += num

avePest_perImg_pestExist = numPest/numImg_pestExist
# average pest number of image that pest exists = 1.785

# %%
insect_count = []
for i,insect_ele in enumerate(train_ann_dic['categories']):
    pest_dir = os.path.join(data_root_dir,str(insect_ele['id']))
    insect_count.append(len(os.listdir(pest_dir)))
np.array(insect_count)
# %%
sample_mul = np.divide(max(insect_count), insect_count)
# max(insect_count)
sample_scale = np.log(sample_mul)
sample_scale = np.ceil(sample_scale + np.divide(sample_mul, 3) - 1)
sample_scale = sample_scale.astype(int)
sampleCount_aferScale = insect_count + np.multiply(insect_count, sample_scale)
sampleCount_aferScale = sampleCount_aferScale.astype(int)
numSample_before = sum(insect_count[1:])
numSample_after = sum(sampleCount_aferScale[1:])
print('sampleCount_aferScale: ', sampleCount_aferScale)
print('numSample_before: ', numSample_before)
print('numSample_after: ', numSample_after)
# %%
test_list_111 = []
test_dic_111 = {'a':1,'b':2}
test_list_111.append(test_dic_111)
test_list_111 += [test_dic_111] * 3

# %%
cropped_pests = []

for i,insect_ele in enumerate(train_ann_dic['categories']):
    pest_dir = os.path.join(data_root_dir,str(insect_ele['id']))
    if insect_ele['id'] == 1:
        continue
    for pest_image_name in os.listdir(pest_dir):
        pest_path = os.path.join(pest_dir, pest_image_name)
        cropped_pest = {
            'category_id': insect_ele['id'],
            'file_path': pest_path
        }
        cropped_pests += [cropped_pest] * sample_scale[insect_ele['id']-1]



# %%
import random


random.seed(2022)
random.shuffle(cropped_pests)

# %%
train_ann_dic['images'][0]

# %%
3648 / 3
# %%
5472 / 3

# %%
blank_images = []

for i,ele in enumerate(numPest_per_imageId):
    if ele == 0:
        image_path = os.path.join(img_dir,train_ann_dic['images'][images_index[i]]['file_name'])
        image_height = train_ann_dic['images'][images_index[i]]['height']
        image_width = train_ann_dic['images'][images_index[i]]['width']

        blank_image = {
            'image_id': i,
            'file_path': image_path,
            'height': image_height,
            'width': image_width
        }
        blank_images += [blank_image]


# %%
def paste_location(bigSize, smallSize):
    # [H,W],[H,W] -> left top [x,y]
    x_range = bigSize[1] - smallSize[1]
    y_range = bigSize[0] - smallSize[0]
    tlX = random.randint(0,x_range)
    tlY = random.randint(0,y_range)
    return [tlX, tlY]

# %%

x1, y1 = paste_location([1200,1200],[200,200])
# %%
# mkdir mmdetection/data/pest-coco/AUG/pasted/
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

# %%
# test
blank_im = Image.open(blank_images[0]['file_path'])
blank_pix = np.array(blank_im)
# blank_pix.shape
blank_pix = blank_pix[::3,::3,:]
blank_pix.shape[0]

# %%
# test
assert(2>1 and 3>2)

# %%
# ran once
pasted_im_dir = 'mmdetection/data/pest-coco/AUG/pasted/'
pop_blank = 1

while len(cropped_pests) and len(blank_images):
    if pop_blank % 2:
        blank_image = blank_images.pop()
        blank_im = Image.open(blank_image['file_path'])
        blank_pix = np.array(blank_im)
        # rescale
        blank_pix = blank_pix[::3,::3,:]
        blank_image['height'] = blank_pix.shape[0]
        blank_image['width'] = blank_pix.shape[1]
    pop_blank += 1

    cropped_pest = cropped_pests.pop()
    cropped_im = Image.open(cropped_pest['file_path'])
    cropped_pix = np.array(cropped_im)

    big_size = [blank_image['height'],blank_image['width']]
    small_size = [cropped_pix.shape[0],cropped_pix.shape[1]]
    assert(big_size[0]>small_size[0] and big_size[1]>small_size[1])
    x1, y1 = paste_location(big_size, small_size)
    x2, y2 = x1+small_size[1], y1+small_size[0]
    blank_pix[y1:y2,x1:x2,:] = cropped_pix

    image_id = blank_image['image_id']
    max_len = len(train_ann_dic['annotations'])
    max_anno_id = train_ann_dic['annotations'][max_len-1]['id']
    anno_id = max_anno_id + 1
    category_id = cropped_pest['category_id']
    bbox = [x1, y1, x2-x1, y2-y1]
    area = bbox[2] * bbox[3]
    iscrowd = 0
    tmp_anno = {
        'id': anno_id,
        'image_id': image_id,
        'category_id': category_id,
        'bbox': bbox,
        'area': area,
        'iscrowd': iscrowd
    }
    train_ann_dic['annotations'].append(tmp_anno)
    if pop_blank % 2:
        train_ann_dic['images'][images_index[image_id]]['height'] = big_size[0]
        train_ann_dic['images'][images_index[image_id]]['width'] = big_size[1]
        image_name = train_ann_dic['images'][images_index[image_id]]['file_name']
        pasted_path = os.path.join(pasted_im_dir, image_name)
        pasted_im = Image.fromarray(blank_pix)
        pasted_im.save(pasted_path)

    
# %%
3 % 2
# %%
import data_utils_v2

# mkdir 'mmdetection/data/pest-coco/AUG/annotations/'
train_ann_file = 'mmdetection/data/pest-coco/AUG/annotations/train_labels.json'
data_utils_v2.save_to_json(train_ann_dic, train_ann_file)

# %%
