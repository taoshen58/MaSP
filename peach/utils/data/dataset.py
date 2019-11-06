from abc import ABCMeta, abstractmethod
import random


class DatasetTemplate(metaclass=ABCMeta):
    def __init__(
            self, data_file_path, data_type, hparams):
        raise NotImplementedError


def data_queue(data, _batch_size, _max_step, *args, **kwargs):
    assert isinstance(_max_step, int)
    assert len(data) >= _batch_size
    len_fn = kwargs.get("len_fn") or len
    random.shuffle(data)
    data_round = 0
    data_ptr, idx_b = 0, 0
    step = 0
    while True:
        batch_buff = []
        while len_fn(batch_buff) < _batch_size:
            if data_ptr >= len(data):  # new data
                random.shuffle(data)
                data_round += 1
                data_ptr, idx_b = 0, 0
            batch_buff.append(data[data_ptr])
            data_ptr += 1
        yield batch_buff, data_round, idx_b
        idx_b += 1
        step += 1
        if step >= _max_step:  # stop condition
            break
    # ===== lagacy code =====
    # data_ptr = 0
    # dataRound = 0
    # idx_b = 0
    # step = 0
    # while True:
    #     if data_ptr + _batch_size <= len(data):
    #         yield data[data_ptr:data_ptr + _batch_size], dataRound, idx_b
    #         data_ptr += _batch_size
    #         idx_b += 1
    #         step += 1
    #     elif data_ptr + _batch_size > len(data):
    #         offset = data_ptr + _batch_size - len(data)
    #         out = data[data_ptr:]
    #         random.shuffle(data)
    #         out += data[:offset]
    #         data_ptr = offset
    #         dataRound += 1
    #         yield out, dataRound, 0
    #         idx_b = 1
    #         step += 1
    #     if step >= _max_step:
    #         break


def data_traverse(data, _batch_size, *args, **kwargs):  # fixme: the dynamic batch size lead to inaccurate batch num
    max_sequence_len = kwargs.get("max_sequence_len")
    len_key = kwargs.get("len_key")

    def _len_batch(_sample_batch):
        return max([len_key(sample) for sample in _sample_batch])

    def _approx_memory(_len, _bs):
        return (_len**2) * (_bs**0.5)

    is_limitied = (max_sequence_len is not None and len_key is not None)
    memory_threshold = _approx_memory(max_sequence_len, _batch_size) if is_limitied else None

    idx_b = 0
    sample_batch = []
    for sample in data:
        sample_batch.append(sample)
        if len(sample_batch) == _batch_size or (
                is_limitied and _approx_memory(_len_batch(sample_batch), len(sample_batch)) >= memory_threshold):
            yield sample_batch, 0, idx_b
            idx_b += 1
            sample_batch = []
    if len(sample_batch) > 0:
        yield sample_batch, 0, idx_b


def data_traverse_lagacy(data, _batch_size, *args, **kwargs):
    idx_b = 0
    sample_batch = []
    for sample in data:
        sample_batch.append(sample)
        if len(sample_batch) == _batch_size:
            yield sample_batch, 0, idx_b
            idx_b += 1
            sample_batch = []
    if len(sample_batch) > 0:
        yield sample_batch, 0, idx_b

