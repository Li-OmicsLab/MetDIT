from __future__ import print_function
import argparse
import os
import cv2
import logging
import torch
import torch.nn.parallel
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from models.resnet import ResNet18
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from plot_metric.functions import BinaryClassification
import warnings

# warnings.filterwarnings("ignore", "is_categorical_dtype")
# warnings.filterwarnings("ignore", "use_inf_as_na")
warnings.filterwarnings("ignore")

def parser_args():
    parser = argparse.ArgumentParser(description='MetDIT Training & Testing via PyTorch')
    parser.add_argument('-dp', '--dataset_path', type=str,
                        default='./evaluation_file/eval_images',
                        help='dataset path for evaluation the MetDIT.')
    parser.add_argument('-gt', '--ground_truth_path', type=str,
                        default='./evaluation_file/ground_truth.txt',
                        help='record of ground truth for all the test data.')
    parser.add_argument('-c', '--checkpoint',
                        default='./evaluation_file/60.pth.tar',
                        type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--use_cuda', default=True, action='store_true',
                        help='Use CUDA to train model')
    parser.add_argument('-m', '--model_name',
                        default='ResNet18', type=str,
                        help='the model used for evaluation')
    parser.add_argument('-sp', '--save_path', default='pred_rec',
                        type=str, help='save record folder for prediction result.')
    parser.add_argument('-vis', '--visual')

    args = parser.parse_args()
    return args


def cal_metric(y_true, y_pred):
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prec = precision_score(y_true=y_true, y_pred=y_pred)
    rec = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)

    return acc, prec, rec, f1


def plot_roc(y_true, y_pred, labels=None):

    if labels is None:
        labels = ["Class 0", "Class 1"]

    bc = BinaryClassification(y_true, y_pred, labels=labels)
    plt.figure(figsize=(15, 10))
    plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
    bc.plot_roc_curve()
    plt.subplot2grid((2, 6), (0, 2), colspan=2)
    bc.plot_precision_recall_curve()
    plt.subplot2grid((2, 6), (0, 4), colspan=2)
    bc.plot_class_distribution()
    plt.subplot2grid((2, 6), (1, 1), colspan=2)
    bc.plot_confusion_matrix()
    plt.subplot2grid((2, 6), (1, 3), colspan=2)
    bc.plot_confusion_matrix(normalize=True)
    # Save figure
    plt.savefig('./evaluation_file/classification_metric.png')

    param_pr_plot = {
        'c_pr_curve': 'blue',
        'c_mean_prec': 'cyan',
        'c_thresh_lines': 'red',
        'c_f1_iso': 'green',
        'beta': 2,
    }

    plt.figure(figsize=(6, 6))
    bc.plot_precision_recall_curve(**param_pr_plot)

    # Save figure
    plt.savefig('./evaluation_file/classification_PRCurve.png')



def evaluation(args):
    # assert os.path.exists(checkpoint_path)

    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))['state_dict']
    if args.model_name == 'ResNet18':
        model = ResNet18(num_classes=2)
        model.load_state_dict(checkpoint)
    else:
        raise NotImplementedError('Only support ResNet18 now.')

    if args.use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda:0')
        logging.info('==> Model training on GPU')
    else:
        device = torch.device('cpu')
        logging.info('==> Model training on CPU')

    model = model.eval()
    model = model.to(device)

    pred_list = []
    gt_list = []
    pred_auc_list = []

    gt_rec = open(args.ground_truth_path, 'r')
    gt_mapping = {}
    for line in gt_rec:
        items = line.strip('\n').split('\t')
        gt_mapping[items[0]] = items[1]

    gt_rec.close()

    pred_rec = open('./evaluation_file/pred_res.csv', 'w')
    save_line = 'name,gt_label,pred_label,pred_score\n'
    pred_rec.write(save_line)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    for img_name in tqdm(os.listdir(args.dataset_path)):
        img_path = os.path.join(args.dataset_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform_test(img).to(device)

        outputs = model(img_tensor.unsqueeze(0))
        outputs = torch.softmax(outputs, dim=1)
        pred = outputs.cpu().detach().numpy()[0]
        # class_prob, top_class = np.max(pred, dim=1)
        label = np.argmax(pred)
        score = pred[label]

        if label == 1:
            pred_auc_list.append(score)
        else:
            pred_auc_list.append(1 - score)

        pred_list.append(label)
        gt_list.append(int(gt_mapping.get(img_name)))

        save_line = '{},{},{},{}\n'.format(img_name, gt_mapping.get(img_name), str(label), str(score))
        pred_rec.write(save_line)

    pred_rec.close()

    acc, prec, rec, f1 = cal_metric(y_true=gt_list, y_pred=pred_list)

    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(prec))
    print('Recall: {}'.format(rec))
    print('F1-Score: {}'.format(f1))

    # plot roc
    plot_roc(y_true=gt_list, y_pred=pred_auc_list)


if __name__ == '__main__':
    args = parser_args()
    evaluation(args)
