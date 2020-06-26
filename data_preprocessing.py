#-*- coding: utf-8 -*-
# Reference: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/build_dataset.py

import argparse
import random
import os

from PIL import Image
from tqdm import tqdm

SIZE = 256
TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
IMAGE_FILE = '.PNG'
random.seed(230)
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/home/nlplab/v2_Development/yoonseok/yoon_proj/raw_data',
                    help="Directory with the original dataset")

parser.add_argument('--split_data_dir', default='/home/nlplab/v2_Development/yoonseok/yoon_proj/preprocessed_data',
                    help="Directory for train,dev,test")


parser.add_argument('--output_dir', default='/home/nlplab/v2_Development/yoonseok/yoon_proj/data/64x64_img',
                    help="Where to write the new data")

def resize_and_save(filename, output_dir, size=SIZE):
    """
    :param filename: image의 원 저장위치
    :param output_dir: image를 저장할 위치
    :param size: resizing할 이미지 크기
    :return:
    """
    class_dir_path = '/'.join(output_dir.split('/')[:-1])

    if not os.path.exists(class_dir_path):
        os.mkdir(class_dir_path)

    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)

    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)

    image.save(output_dir)

def make_dir(dir_path):
    try:
        if not(os.path.isdir(dir_path)):
            os.makedirs(dir_path)

    except OSError:
        print(" >> Error: Creating directory", dir_path)


args = parser.parse_args()

assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

# Define the data directories
train_data_dir = os.path.join(args.split_data_dir, 'train')
valid_data_dir = os.path.join(args.split_data_dir, 'valid')
test_data_dir = os.path.join(args.split_data_dir, 'test')

all_dict = dict()
all_dict["train"] = []
all_dict["valid"] = []
all_dict["test"] = []


# Scan all the directories in the original dataset
full_dirnames = os.listdir(args.data_dir)
for dirname in full_dirnames:

    # 현재 클래스 디렉토리 내 모든 파일 읽기
    dirpath = os.path.join(args.data_dir, dirname)
    filelist = os.listdir(os.path.join(args.data_dir, dirname))

    # 모든 파일 중 확장자가 '.jpg'인 이미지만 읽어서 이름 순으로 저장하기
    #filenames = [os.path.join(train_data_dir, f) for f in filelist if f.endswith('.jpg')]
    filenames = [os.path.join(dirname, f) for f in filelist if f.endswith(IMAGE_FILE)]

    # 파일 이름순서대로 Sorting & shuffle
    filenames.sort()
    random.shuffle(filenames)

    num_split = int(TRAIN_RATIO * len(filenames))
    num_val_split = int((1-VALID_RATIO) * num_split)
    train_filenames = filenames[:num_val_split]
    valid_filenames = filenames[num_val_split:num_split]
    test_filenames = filenames[num_split:]

    # train/valid/test에 저장될 이미지의 "class이름/이미지파일명" 리스트를 저장
    all_dict["train"].extend(train_filenames)
    all_dict["valid"].extend(valid_filenames)
    all_dict["test"].extend(test_filenames)

if not os.path.exists(args.split_data_dir):
    os.mkdir(args.split_data_dir)
else:
    print("Warning: output dir {} already exists".format(args.split_data_dir))

# Preprocess train, val and test
for split in ['train', 'valid', 'test']:
    split_datapath = os.path.join(args.split_data_dir, '{}'.format(split))
    if not os.path.exists(split_datapath):
        os.mkdir(split_datapath)
    else:
        print("Warning: dir {} already exists".format(split_datapath))

    print("Processing {} data, saving preprocessed data to {}".format(split, split_datapath))
    for filename in tqdm(all_dict[split]):
        original_filename = os.path.join(args.data_dir, filename)
        new_split_datapath = os.path.join(split_datapath, filename)
        resize_and_save(original_filename, new_split_datapath, size=SIZE)

print("Done building dataset")



