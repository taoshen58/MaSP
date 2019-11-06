import numpy as np


class MovingAverageDict(object):
    def __init__(self, decay=0.99):
        self.decay = decay
        self.ma_dict = {}

    def __call__(self, value_dict):
        for key, val in value_dict.items():
            if isinstance(val, (np.float32, np.float64, np.float16)) or \
                    (isinstance(val, np.ndarray) and val.dtype == "float32" and val.ndim == 0):
                val = float(val)

            if isinstance(val, float):
                if key not in self.ma_dict:
                    self.ma_dict[key] = MovingAverage()
                self.ma_dict[key](val)

    def get_val_dict(self):
        dict_return = {}
        for key, ma_obj in self.ma_dict.items():
            dict_return[key] = ma_obj.value
        return dict_return

    def get_val_str(self):
        val_dict = self.get_val_dict()
        # sort
        sorted_list = list(sorted(val_dict.items(), key=lambda item: item[0]))
        str_return = ""
        for key, val in sorted_list:
            if len(str_return) > 0:
                str_return += ", "
            str_return += "%s: %.4f" % (key, val)
        return str_return


class MovingAverage(object):
    def __init__(self, decay=0.99):
        self.decay = decay
        self.value = None

    def __call__(self, new_val):
        if self.value is None:
            self.value = new_val
        else:
            self.value = self.decay * self.value + (1. - self.decay) * new_val
        return self.value
