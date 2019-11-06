from scipy.stats import spearmanr
from scipy.stats import pearsonr

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
            f1 = 2 * precision * recall
        else:
            f1 = 0.
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    else:
        raise NotImplementedError


def data_correlation(list_a, list_b, method):
    if method == "spearman":
        return spearmanr(list_a, list_b)
    elif method == "pearson":
        return pearsonr(list_a, list_b)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    a = [
        0.5583,
        0.5727,
        0.6122,
        0.5553,
        0.6096,
        0.6180,
        0.6417,
        0.5117,
        0.6251,
    ]
    b1 = [
        5.736,
        5.507,
        5.100,
        7.150,
        6.233,
        8.318,
        7.928,
        5.272,
        8.490,
    ]
    b2 = [
        9.030,
        8.453,
        6.095,
        10.298,
        7.033,
        10.094,
        9.285,
        8.025,
        14.415,
    ]
    b = [e1/e2 for e1, e2 in zip(b1, b2)]

    print(
        "SpearmanR: {}; PearsonR: {}".format(
            data_correlation(a, b, "spearman"),
            data_correlation(a, b, "pearson"),
        )
    )

class F1MetricV2(object):
    TEMPLATE = {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0,
    }
    def __init__(self):
        self.counter = self.TEMPLATE.copy()

    def clear(self):
        self.counter = self.TEMPLATE.copy()

    def add(self, label, pred):
        if label > 0:
            if label == pred:
                self.counter["TP"] += 1
            else:
                self.counter["FN"] += 1
        else:
            if label == pred:
                self.counter["TN"] += 1
            else:
                self.counter["FP"] += 1

    def add_list(self, iter):
        for label, pred in iter:
            self.add(label, pred)

    def get_metric(self, name=None):
        precision = 1. * self.counter["TP"] / (self.counter["TP"] + self.counter["FP"]) if self.counter["TP"] > 0 else 0.
        recall = 1. * self.counter["TP"] / (self.counter["TP"] + self.counter["FN"]) if self.counter["TP"] > 0 else 0.
        f1_score = 2. * precision * recall / (precision + recall) if (precision + recall) > 0. else 0.
        accuracy = 1. * (self.counter["TP"] + self.counter["TN"]) / (
                self.counter["TP"] + self.counter["TN"] + self.counter["FP"] + self.counter["FN"]) if (self.counter["TP"] + self.counter["TN"]) > 0. else 0.
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


class F1MetricV3(object):  # no accuracy
    TEMPLATE = {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0,
    }
    def __init__(self):
        self.counter = self.TEMPLATE.copy()

    def clear(self):
        self.counter = self.TEMPLATE.copy()

    def add(self, predicted, labeled):
        pl = set(predicted)
        ll = set(labeled)

        self.counter["TP"] += len(pl & ll)
        self.counter["FP"] += len(pl - ll)
        self.counter["FN"] += len(ll - pl)
        self.counter["TN"] += 0

    def add_list(self, iter):
        for label, pred in iter:
            self.add(label, pred)

    def get_metric(self, name=None):
        precision = 1. * self.counter["TP"] / (self.counter["TP"] + self.counter["FP"]) if self.counter["TP"] > 0 else 0.
        recall = 1. * self.counter["TP"] / (self.counter["TP"] + self.counter["FN"]) if self.counter["TP"] > 0 else 0.
        f1_score = 2. * precision * recall / (precision + recall) if (precision + recall) > 0. else 0.
        return_dict = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }
        if isinstance(name, str) and len(name) > 0:
            new_return_dict = {}
            for key, val in return_dict.items():
                new_return_dict[name + "_" + key] = val
            return_dict = new_return_dict
        return return_dict





