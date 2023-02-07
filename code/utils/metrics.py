
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def get_confusionmatrix_fnd(preds, labels):
    # label_predicted = np.argmax(preds, axis=1)
    label_predicted = preds
    print (accuracy_score(labels, label_predicted))
    print(classification_report(labels, label_predicted, labels=[0.0, 1.0], target_names=['real', 'fake'],digits=4))
    print (confusion_matrix(labels, label_predicted, labels=[0,1]))


def metrics(y_true, y_pred):

    metrics = {}
    metrics['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    y_pred = np.around(np.array(y_pred)).astype(int)
    metrics['f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    metrics['acc'] = accuracy_score(y_true, y_pred)

    return metrics