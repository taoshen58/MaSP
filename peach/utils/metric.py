import numpy as np


class F1ScoreMetric(object):
    def __init__(self):
        #                   precision
        #                     ||
        # True Negative, False Positive
        # False Negative, True Positive ==>recall
        self.counter = np.zeros([2, 2],  "int32")

    def clear(self):
        self.counter = np.zeros([2, 2], "int32")

    def add(self, label, pred):
        self.counter[int(label), int(pred)] += 1

    def add_list(self, iter):
        for label, pred in iter:
            self.add(label, pred)

    def get_metric(self, name=None):
        precision = 1. * self.counter[1, 1] / np.sum(self.counter[:, 1])  # (self.counter[1, 1] + self.counter[0, 1])
        recall = 1. * self.counter[1, 1] / np.sum(self.counter[1, :])  # (self.counter[1, 1] + self.counter[1, 0])
        f1_score = 2 * precision * recall / (precision + recall)
        accuracy = (self.counter[1, 1] + self.counter[0, 0]) / np.sum(self.counter)
        return_dict = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
        }
        if isinstance(name, str) and len(name) > 0:
            new_return_dict = {}
            for key, val in return_dict.items():
                new_return_dict[name + "_" + key] = val
            return_dict = new_return_dict
        return return_dict


def f1_metric(predicted, labeled):
    if isinstance(predicted[0], (list, set)):
        counter = {
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "TN": 0,
        }

        for pl, ll in zip(predicted, labeled):
            pl = set(pl)
            ll = set(ll)

            counter["TP"] += len(pl & ll)
            counter["FP"] += len(pl - ll)
            counter["FN"] += len(ll - pl)
            counter["TN"] += 0

        precision = counter["TP"] / (counter["TP"] + counter["FP"])
        recall = counter["TP"] / (counter["TP"] + counter["FN"])

        if precision > 0. and recall > 0.:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    else:
        raise NotImplementedError










