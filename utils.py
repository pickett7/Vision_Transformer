import os
import math
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
import pandas as pd

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']


def encode_labels(target):
    
    ls = target['annotation']['object']
  
    j = []
    if type(ls) == dict:
        if int(ls['difficult']) == 0:
            j.append(object_categories.index(ls['name']))
  
    else:
        for i in range(len(ls)):
            if int(ls[i]['difficult']) == 0:
                j.append(object_categories.index(ls[i]['name']))
    
    k = np.zeros(len(object_categories))
    k[j] = 1
  
    return torch.from_numpy(k)


def plot_history(train_hist, val_hist, y_label, filename, labels=["train", "validation"]):

    # Plot loss and accuracy
    xi = [i for i in range(0, len(train_hist), 2)]
    plt.plot(train_hist, label = labels[0])
    plt.plot(val_hist, label = labels[1])
    plt.xticks(xi)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.savefig(filename)
    plt.show()


def get_ap_score(y_true, y_scores):

    scores = 0.0
    
    for i in range(y_true.shape[0]):
        scores += average_precision_score(y_true = y_true[i], y_score = y_scores[i])
    
    return scores

def save_results(images, scores, columns, filename):

    df_scores = pd.DataFrame(scores, columns=columns)
    df_scores['image'] = images
    df_scores.set_index('image', inplace=True)
    df_scores.to_csv(filename)


def append_gt(gt_csv_path, scores_csv_path, store_filename):

    gt_df = pd.read_csv(gt_csv_path)
    scores_df = pd.read_csv(scores_csv_path)
    
    gt_label_list = []
    for index, row in gt_df.iterrows():
        arr = np.array(gt_df.iloc[index,1:], dtype=int)
        target_idx = np.ravel(np.where(arr == 1))
        j = [object_categories[i] for i in target_idx]
        gt_label_list.append(j)
    
    scores_df.insert(1, "gt", gt_label_list)
    scores_df.to_csv(store_filename, index=False)