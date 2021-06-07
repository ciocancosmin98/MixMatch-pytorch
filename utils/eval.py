from __future__ import print_function, absolute_import
from torchvision.transforms import ToTensor
import PIL.Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import io

__all__ = ['accuracy', 'confusion_matrix', 'plot_confusion_matrix', 'precision_recall']

def accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1)
    pred = pred.view(-1)
    
    targets = targets.long()
    correct = pred.eq(targets)

    result = correct.float().sum(0).mul_(100 / batch_size)

    return result

def confusion_matrix(outputs, targets):
    batch_size = targets.size(0)
    n_classes  = outputs.size(1)

    _, pred = outputs.topk(1, 1)
    pred = pred.view(-1)
    
    targets = targets.long()

    confusion = torch.zeros(n_classes, n_classes)
    for i in range(batch_size):
        confusion[targets[i]][pred[i]] += 1

    return confusion

def precision_recall(confusion):
    precision = []
    recall    = []

    confusion = confusion.numpy()

    n_classes = confusion.shape[0]

    eps = 0.0000001

    for i in range(n_classes):
        sum_row = np.sum(confusion[i, :])
        sum_col = np.sum(confusion[:, i])

        rec = confusion[i][i] / sum_row if abs(sum_row) > eps else 0.0
        pre = confusion[i][i] / sum_col if abs(sum_col) > eps else 0.0

        precision.append(pre)
        recall.append(rec)

    return precision, recall

# https://discuss.pytorch.org/t/example-code-to-put-matplotlib-graph-to-tensorboard-x/15806
def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)

    shp = image.shape
    image = image.reshape((shp[1], shp[2], shp[3]))

    return image

# https://www.youtube.com/watch?v=k7KfYXXrOj0&ab_channel=AladdinPersson
def plot_confusion_matrix(confusion, class_names):
    confusion=confusion.numpy()

    n_classes = confusion.shape[0]
    figure = plt.figure(figsize=(n_classes, n_classes))
    plt.imshow(confusion, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")

    indices = np.arange(n_classes)
    plt.xticks(indices, class_names, rotation=45)
    plt.yticks(indices, class_names)

    confusion = np.around(
        confusion / confusion.sum(axis=1)[:, np.newaxis], decimals=3
    )

    threshold = confusion.max() / 2.0

    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if confusion[j, i] > threshold else "black"

            plt.text(
                i, j, confusion[j, i], horizontalalignment="center", color=color
            )

    plt.tight_layout(pad=2)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    image = plot_to_image(figure)
    return image