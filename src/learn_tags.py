#!/usr/bin/env python3

from pathlib import Path
import random
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from matplotlib import pyplot as plt
from arc2020 import io, visualization, solve, solver, main, pretrain
from arc2020.solver.ops.learnable.cnn import learn_tags


def aggregate_task(predictor, task):
    preds = []
    for img_pair in task.train:
        preds.append(predictor(img_pair))
    return np.mean(preds, axis=0)


if __name__ == '__main__':
    data_path = Path('../data/training')
    all_tasks = io.read_all_tasks(data_path)
    io.load_tags(all_tasks, Path('../experiments/training_tasks_tagged.csv'))
    all_keys = list(all_tasks.keys())
    tags_names = [tag_name for tag_name, tag_val in all_tasks[all_keys[0]].tags]
    random.shuffle(all_keys)
    train_val = 0.9
    train_keys = all_keys[:int(train_val * len(all_keys))]
    test_keys = all_keys[int(train_val * len(all_keys)):]
    train_tasks = {key: all_tasks[key] for key in train_keys}
    test_tasks = {key: all_tasks[key] for key in test_keys}
    predictor = learn_tags(train_tasks)
    test_preds = np.array([aggregate_task(predictor, cur_task) for cur_task in test_tasks.values()])
    test_tags = np.array([[tag_val for tag_name, tag_val in cur_task.tags] for cur_task in test_tasks.values()])
    test_acc = [[balanced_accuracy_score(cur_tag, cur_pred > threshold)
                 for cur_tag, cur_pred in zip(test_tags.T, test_preds.T)]
                for threshold in np.arange(0.1, 0.9, 0.1)]
    test_roc = [roc_auc_score(cur_tag, cur_pred) if (len(np.unique(cur_tag)) > 1) else -1
                for cur_tag, cur_pred in zip(test_tags.T, test_preds.T)
                ]
    with open('../experiments/res.csv', 'w') as f:
        print(','.join(tags_names), file=f)
        print(','.join([f'{idx}' for idx in range(len(tags_names))]), file=f)
        print(','.join([f'{cur_auc:.4f}' for cur_auc in test_roc]), file=f)
        for cur_acc in test_acc:
            print(','.join([f'{tag_acc:.5f}' for tag_acc in cur_acc]), file=f)
    print('Done!')
