import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)

        loss = -((1 - pt) ** self.gamma) * logpt
        if self.alpha is not None:
            at = self.alpha.gather(0, target)
            loss = loss * at

        return loss.mean() if self.reduction == 'mean' else loss.sum()


# Define Confusion Matrix Class
class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    # Update Matrix.
    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # Calculate overall accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("The model accuracy is {:.4f}".format(acc))

        # Precision, Recall, Specificity, F1
        from prettytable import PrettyTable
        table = PrettyTable()
        table.field_names = ["Class", "Precision", "Recall", "Specificity", "F1-score"]

        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = TP / (TP + FP) if TP + FP != 0 else 0.
            Recall = TP / (TP + FN) if TP + FN != 0 else 0.
            Specificity = TN / (TN + FP) if TN + FP != 0 else 0.
            F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) != 0 else 0.
            table.add_row([
                self.labels[i],
                round(Precision, 3),
                round(Recall, 3),
                round(Specificity, 3),
                round(F1, 3)
            ])
        print(table)

    def plot(self, figsize=(8, 6), save_path=None, annot_fontsize=14, cmap="Blues", fmt='d'):
        """
        Plot the confusion matrix using seaborn heatmap.
        """
        matrix = self.matrix.astype(int)
        plt.figure(figsize=figsize)
        ax = sns.heatmap(matrix,
                        annot=True,
                        fmt=fmt,
                        cmap=cmap,
                        cbar=True,
                        linewidths=0.5,
                        square=True,
                        xticklabels=self.labels,
                        yticklabels=self.labels,
                        annot_kws={"size": annot_fontsize, "weight": "bold"})
        plt.xlabel('True Labels', fontsize=15)
        plt.ylabel('Predicted Labels', fontsize=15)
        plt.title('Confusion Matrix', fontsize=17)
        plt.xticks(rotation=45, ha="right", fontsize=13)
        plt.yticks(rotation=0, fontsize=13)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()
        plt.close()
