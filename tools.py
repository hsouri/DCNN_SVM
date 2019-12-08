import numpy as np
from matplotlib import pyplot as plt
import itertools


def confusion_matrix(real_labels, predicted_labels, num_of_classes, out_file):
    confusion = np.zeros((num_of_classes, num_of_classes))
    for i in range(real_labels.shape[0]):
        confusion[real_labels[i], predicted_labels[i]] += 1
    return confusion

def plot_confusion_matrix(confusion_matrix, num_of_classes, plot_name, normalize=True, cmap=plt.cm.Blues):
    if normalize:
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(plot_name + " confusion matrix")
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(plot_name + " confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(num_of_classes)
    plt.xticks(tick_marks, np.arange(10))
    plt.yticks(tick_marks, np.arange(10))

    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(plot_name + " confusion matrix" + ".jpg")


def write_results(accuracy, file, name):
    f = open(file, 'a')
    f.write('\n' + name + ' accuracy is ' + str(accuracy) + '\r\n')
    f.close()