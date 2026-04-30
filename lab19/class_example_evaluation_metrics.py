import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def evaluation_metrics():
    X = np.array([[0.85], [0.6], [0.7], [0.4], [0.55], [0.5], [0.65], [0.35], [0.6], [0.2]])
    y = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])

    threshold = 0.5

    """To get predicted labels"""
    predicted_label = []
    for i in X:
        if i[0] >= threshold:
            predicted_label.append(1)
        else:
            predicted_label.append(0)

    predicted_label = np.array(predicted_label)
    print("Predicted labels: ", predicted_label)

    tp = np.sum((predicted_label == 1) & (y == 1))
    tn = np.sum((predicted_label == 0) & (y == 0))
    fp = np.sum((predicted_label == 1) & (y == 0))
    fn = np.sum((predicted_label == 0) & (y == 1))


    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print("-----------------------------------")
    print("Accuracy: ", accuracy)

    precision = tp / (tp + fp)
    print("-----------------------------------")
    print("Precision: ", precision)
    print("-----------------------------------")
    recall = tp / (tp + fn)
    print("recall: ", recall)
    print("-----------------------------------")

    specificity = tn / (tn + fp)
    print("Specificity: ",specificity)
    print("-----------------------------------")
    f1_score = 2 * precision * recall / (precision + recall)
    print("F1-score: ", f1_score)


def main():

    evaluation_metrics()

if __name__ == "__main__":
    main()
