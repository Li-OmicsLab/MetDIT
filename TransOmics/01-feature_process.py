import os
import random
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from tqdm import tqdm
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")


def parser_args():
    parser = argparse.ArgumentParser(description='Data Pre-process for TransOmics with Python')
    parser.add_argument('-ofp', '--original_file_path', type=str,
                        default='./demo_file/human_cachexia_dataset.csv',
                        help='original csv path of input file')
    parser.add_argument('-sfp', '--save_file_path', type=str,
                        default='./demo_result/save_record_01',
                        help='save post-processing file path')
    parser.add_argument('-sn', '--save_file_name', type=str,
                        default='demo',
                        help='save post-processing file name')
    parser.add_argument('-r', '--rate', type=float,
                        default=0.2,
                        help='the threshold of none or null items for drop this features')
    parser.add_argument('-vis', '--visualization',
                        action='store_true',
                        help='the results of feature ranking visualization')
    parser.add_argument('-vn', '--vis_num', type=int,
                        default=10,
                        help='the number of reserved feature ranking')
    parser.add_argument('-log', '--log_func', action='store_true',
                        help='use or not the log_function for data normalization, the default is true')
    parser.add_argument('-norm', '--norm_func',
                        choices=['minmax', 'zscore', 'maxabs', 'l2', 'l1'],
                        default='minmax',
                        help='use or not the z-score for data normalization, the default is false')
    parser.add_argument('-mt', '--method_type',
                        choices=['RF', 'DT', 'XGB', 'LGB', 'CB'],
                        default='RF',
                        help='the method used for feature ranking.')

    args = parser.parse_args()

    return args


def load_csv(csv_path, rate=0.2):
    """
    Load csv file and processing the null or black values.
    """

    assert os.path.exists(csv_path)

    frames = pd.read_csv(csv_path, index_col=0)
    feature_name = frames.columns
    label_list = frames.index

    frames = frames.T
    total_num = frames.shape[1]
    save_title = frames.columns.values[:]
    save_title_new = [str(int(ele)) for ele in save_title]

    frames = frames.values
    frames_new = []

    for idx in range(len(frames)):
        # feature_name = 'Feature_' + str(frames[idx][0])
        vals = frames[idx]
        nan_sum = np.sum(np.isnan(vals))
        nan_list = np.isnan(vals)
        nan_list = np.where(nan_list, 1.0, 0.0)
        if nan_sum >= total_num * rate:  # select feature
            continue
        else:
            item = np.nanmin(vals) * 0.2
            nan_list *= item
            vals = np.where(np.isnan(vals), 0, vals)
            vals += nan_list

        vals = vals.tolist()
        # vals.insert(0, save_title_new[idx])
        frames_new.append(vals)

    frames_new = pd.DataFrame(frames_new)
    frames_new = frames_new.T
    frames_new.columns = feature_name
    frames_new.index = label_list

    return frames_new


def feature_ranking(frames, method_type='RF', file_name='sample', vis=True, vis_num=None, args=None):
    """
    feature ranking by ML function.
    Random Forest is the default function for feature ranking.
    """

    if frames is None:
        raise ValueError('Input sample is None.')

    # bio-marker name
    feature_name = frames.columns.values
    # data
    X_data = frames.values
    # label
    Y_data = frames.index.values.astype(int)

    if method_type == 'RF':
        print('==> Using Random Forest (RF) for feature important ranking.')
        clf = RandomForestClassifier(random_state=42)

    elif method_type == 'LGB':
        print('==> Using LightGBM (LGB) for feature important ranking.')
        # clf = lgb.LGBMClassifier()
        clf = lgb.LGBMClassifier(
            boosting_type='gbdt', num_leaves=55, reg_alpha=0.0, reg_lambda=1,
            max_depth=15, n_estimators=6000, objective='binary',
            subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
            learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4, verbose=-1
        )

    elif method_type == 'XGB':
        print('==> Using XGBoost (XGB) for feature important ranking.')
        clf = XGBClassifier(seed=42)

    elif method_type == 'DT':
        print('==> Using Decision TREE (DT) for feature important ranking.')
        clf = DecisionTreeClassifier(random_state=42)

    elif method_type == 'CB':
        print('==> Using CatBoost (CB) for feature important ranking.')
        clf = CatBoostClassifier()

    else:
        raise NotImplementedError('Feature important ranking not support {} now.'.format(method_type))

    # model fit
    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=10)
    clf.fit(X_train, y_train)

    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, feature_name)), columns=['Value', 'Feature'])

    # setting feature visualization number
    if vis_num is None:
        vis_num = 15
    elif vis_num < 0:
        raise ValueError('Feature Visualization Number should > 0')
    elif vis_num > X_data.shape[1]:
        raise ValueError('Feature Visualization Number is larger than the total feature number of input samples.')

    if vis:
        plt.figure(figsize=(40, 20))
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[:vis_num])
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        vis_save_path = os.path.join(args.save_file_path,
                                     '{}_importances_{}.png'.format(method_type, file_name))
        plt.savefig(vis_save_path)
        plt.show()

    feature_imp = feature_imp.sort_values(by="Value", ascending=False)
    rec_save_path = os.path.join(args.save_file_path,
                                 '{}_importances_{}.csv'.format(method_type, file_name))
    feature_imp.to_csv(rec_save_path, index=False)

    selected_feature = feature_imp.sort_values(by="Value", ascending=False)[:vis_num]
    selected_feature_name = selected_feature['Feature'].values

    print('==> Select top {} features.'.format(vis_num))
    # print('==> Top {} feature names as follow: '.format(vis_num))
    # vis_line = ', '.join(selected_feature_name)
    # print(vis_line)

    return selected_feature_name


def select_feature_by_name(frames, top_feature_names):
    """
    select the top feature by ranked feature name.
    """

    assert frames is not None
    assert top_feature_names is not None
    top_feature_names = top_feature_names.tolist()
    selected_frames = frames[top_feature_names]

    return selected_frames


def data_normalization(frames, log_func=True, norm_func='minmax'):
    """
    data normalization by log and z-score.
    """

    assert frames is not None
    if log_func:
        norm_res = frames.map(np.log2)
    else:
        norm_res = frames

    # if addition data normalization:
    if norm_func == 'minmax':
        norm_res = norm_res.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    elif norm_func == 'zscore':
        norm_res = norm_res.apply(lambda x: (x - x.mean()) / (x.std()))

    elif norm_func == 'maxabs':
        max_abs_scaler = preprocessing.MaxAbsScaler().fit(norm_res)
        norm_res_sk = max_abs_scaler.transform(norm_res)
        norm_res_sk = pd.DataFrame(norm_res_sk)
        norm_res_sk.index = norm_res.index
        norm_res_sk.columns = norm_res.columns
        norm_res = norm_res_sk

    elif norm_func == 'l2':
        norm_res_sk = preprocessing.normalize(norm_res, norm='l2')
        norm_res_sk = pd.DataFrame(norm_res_sk)
        norm_res_sk.index = norm_res.index
        norm_res_sk.columns = norm_res.columns
        norm_res = norm_res_sk

    elif norm_func == 'l1':
        norm_res_sk = preprocessing.normalize(norm_res, norm='l1')
        norm_res_sk = pd.DataFrame(norm_res_sk)
        norm_res_sk.index = norm_res.index
        norm_res_sk.columns = norm_res.columns
        norm_res = norm_res_sk

    else:
        raise NotImplementedError('Not support Normalization of {}'.format(args.norm_func))

    if not log_func and not norm_func:
        return frames

    return norm_res


def save_file(frames, save_path=None, name='demo'):
    """
    save file for image generation.
    """

    assert frames is not None

    # if save_path is None:
    #     raise ValueError('File save path is none. Please check.')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_save_path = os.path.join(save_path, 'post_{}.csv'.format(name))
    frames_trans = frames.T
    # print(frames_trans)
    print('==> Save file to {}'.format(file_save_path))
    frames_trans.to_csv(file_save_path, index=False)


def run_command(args):
    """
        model running
    """
    frames = load_csv(args.original_file_path,
                      args.rate)  # frame@dataframe : horizontal data type, each row defines a ID sample.

    # data normalization
    norm_frames = data_normalization(frames, log_func=args.log_func, norm_func=args.norm_func)

    # make save folder, if it not exists
    if not os.path.exists(args.save_file_path):
        os.makedirs(args.save_file_path)

    # get the dataframe for feature ranking
    top_feature_names = feature_ranking(norm_frames,
                                        method_type=args.method_type,
                                        file_name=args.original_file_path.split('.')[0],
                                        vis=args.visualization,
                                        vis_num=args.vis_num,
                                        args=args)

    # select elements by top_feature_names
    top_norm_frames = select_feature_by_name(norm_frames, top_feature_names)
    top_ori_frames = select_feature_by_name(frames, top_feature_names)

    # save to csv
    save_file(top_norm_frames, save_path=args.save_file_path, name=args.save_file_name)
    save_file(top_ori_frames, save_path=args.save_file_path, name='top_ori_frames')


if __name__ == '__main__':
    args = parser_args()
    run_command(args)

    print('==> Data pre-processing finish.')
