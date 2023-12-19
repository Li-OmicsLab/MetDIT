#!/usr/bin/python
# generate the converted samples from the input csv files
import argparse
import os
import numpy as np
import pandas as pd
from matplotlib import image
from pyts.image import GramianAngularField
from tqdm import tqdm


def parser_args():
    parser = argparse.ArgumentParser(description='TransDIT with Python')
    parser.add_argument('-fn', '--file_name', type=str,
                        default='./demo_result/save_record_01/post_top_ori_frames.csv',
                        help='csv path of sequence file')
    parser.add_argument('-sp', '--save_path', type=str,
                        default='./demo_result/generated_results_01',
                        help='converted images save path')
    parser.add_argument('-sz', '--img_sz', type=int, choices=[8, 16, 32, 64, 128], default=32,
                        help='image resolution size of converted images, only can choose from "[8, 16, 32, 64, 128]"')
    parser.add_argument('-mt', '--method_type', choices=['summation', 'difference'],
                        default='summation',
                        help='converted type of TransDIT, only can choose from "[summation, difference]"')

    args = parser.parse_args()

    return args


def make_save_folder(save_path, label_groups):
    # assert os.path.exists(save_path)

    label_set = set(label_groups)
    img_path = os.path.join(save_path, 'images')  # path of saving converted images
    data_path = os.path.join(save_path, 'data')  # path of saving single data files

    print('==> Converted save path: {} \n\t'
          'Original file save path: {}'.format(img_path, data_path))

    for ele in label_set:
        sub_img_path = os.path.join(img_path, ele)
        sub_data_path = os.path.join(data_path, ele)

        if not os.path.exists(sub_img_path):
            os.makedirs(sub_img_path)
        if not os.path.exists(sub_data_path):
            os.makedirs(sub_data_path)


def extented_feature(frames, img_sz):
    assert frames is not None

    res = frames
    hi = frames.shape[0]
    if hi < img_sz:
        ele = img_sz // hi + 1

        for idx in range(ele):
            res = np.vstack((res, frames))

    return res


def generate_function(method_type, img_sz, save_path, file_name):
    """
    convert the input sequence data into RGB image one.
    """
    print('==> Generate method type: {0} \n\t'
          'Image resolution: {1} * {1}'.format(method_type, img_sz))

    print('==> Start generating images ...')

    src_data = pd.read_csv(file_name)

    img_num = src_data.shape[1]
    label = src_data.columns.values.astype(float).astype(int).astype(str)
    src_data = src_data.values

    # pre-processing dataset
    src_data = extented_feature(src_data, img_sz)

    # make save folder
    make_save_folder(save_path, label)

    gaf = GramianAngularField(image_size=img_sz, method=method_type)
    gaf_images = gaf.fit_transform(src_data.T)

    for idx in tqdm(range(img_num)):
        gaf_img = gaf_images[idx, :, :]
        # img_save_path = os.path.join(img_path, '{}.png'.format(str(idx)))
        # data_save_path = os.path.join(data_path, '{}.csv'.format(str(idx)))
        img_save_path = os.path.join(save_path, 'images', label[idx],
                                     '{}.png'.format(str(idx)))
        image.imsave(img_save_path, gaf_img)

        data_save_path = os.path.join(save_path, 'data', label[idx],
                                      '{}.csv'.format(str(idx)))
        np.savetxt(data_save_path, gaf_img, delimiter=',')

    print("==> Finish! Processing {} images.".format(img_num))


if __name__ == '__main__':
    # get augmentation parameters
    args = parser_args()
    # converted sequence into images
    generate_function(args.method_type, args.img_sz, args.save_path, args.file_name)

