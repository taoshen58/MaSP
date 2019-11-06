import time
import logging
from collections import deque


class TimeEstimator(object):
    def __init__(self, data_limit_len=10):
        self.data_queue = deque(maxlen=data_limit_len)

    def add_data_point(self, completion_rate, logging_et=False):
        data_point = (time.time(), completion_rate)
        if len(self.data_queue) == self.data_queue.maxlen:
            self.data_queue.popleft()
        self.data_queue.append(data_point)
        if logging_et:
            self.logging_estimated_time()

    def logging_estimated_time(self):
        if len(self.data_queue) >=2:
            time_diff = self.data_queue[-1][0] - self.data_queue[0][0]
            comp_diff = self.data_queue[-1][1] - self.data_queue[0][1]

            comp_diff = comp_diff if comp_diff > 0. else 1e-5

            estimated_time = time_diff / comp_diff * (1 - self.data_queue[-1][1])

            et_h = estimated_time // 3600
            et_m = (estimated_time % 3600) // 60

            logging.info("The estimated time to complete training is %dh:%dm" % (et_h, et_m))
        else:
            logging.info("The estimated time to complete training is not available due to limited data")




















