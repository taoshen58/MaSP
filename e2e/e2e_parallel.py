import tensorflow as tf
import numpy as np
import multiprocessing
from os.path import join
import os
import logging
from tqdm import tqdm
from peach.utils.hparams import HParamsCenter, HParams
from peach.utils.file import load_file, save_file
from peach.utils.numeral import MovingAverageDict
from peach.utils.string import get_val_str_from_dict
from peach.tf_nn.utils.graph_handler import GraphHandler, calc_training_step
from peach.tf_nn.utils.performance_recoder import PerformanceRecoder

from utils.csqa import get_data_path_list, load_json, save_json, load_pickle, save_pickle

from e2e.configs import Configs
from e2e.e2e_dataset import E2eDataset, BaseProcessor
from e2e.e2e_evaluator import E2eEvaluator, accumulative_eval, smp_result_print_wrt_qt

from e2e.exe import LfExecutor


def parallel_test(model_cfg, infer_cfg):
    dataset_obj = load_file(infer_cfg['processed_path'] + ".light", 'processed_datasets', mode='pickle')  #
    assert dataset_obj is not None
    dataset_obj._train_feature_list, dataset_obj._dev_feature_list =[], []

    # # processing test data
    # test_feature_list = load_file(infer_cfg['processed_path'] + ".test.pkl", 'test features list', mode='pickle') \
    #     if infer_cfg["load_preproc"] else None
    # if test_feature_list is None or not infer_cfg["load_preproc"]:
    #     test_feature_list = dataset_obj.process_test_data(join(raw_data_dir, "test"))
    #     save_file(test_feature_list, infer_cfg['processed_path'] + ".test.pkl", 'test features list', mode='pickle')

    # assign gpu
    process_idxs, gpu_idxs = assign_gpu_idx(infer_cfg["num_parallels"], infer_cfg["gpu"])
    logging.info("Num of GPUs is {} and num of Processes is {}".format(len(gpu_idxs), len(set(gpu_idxs))))

    # dump dir
    dump_dir = join(infer_cfg["other_dir"], infer_cfg["dump_dir"])
    if os.path.exists(dump_dir) and os.path.isdir(dump_dir):
        if infer_cfg["clear_dump_dir"]:
            path_list_to_rm = get_data_path_list("all", dump_dir)
            for _path_rm in path_list_to_rm:
                os.remove(_path_rm)
    else:
        os.mkdir(dump_dir)

    # get all path list
    all_path_list = get_data_path_list("all", infer_cfg['test_data_path'])
    if infer_cfg["debug_dec"] > 0:
        all_path_list = all_path_list[:infer_cfg["debug_dec"]]

    # # assign to each process
    paths_list = [[] for _ in range(infer_cfg["num_parallels"])]
    for _idx_path, _path in enumerate(all_path_list):
        paths_list[_idx_path % infer_cfg["num_parallels"]].append(_path)

    # check if all exist
    dump_path_list = get_data_path_list("all", dump_dir)
    dump_basename_set = set(os.path.basename(_path) for _path in dump_path_list)
    top1_pred, dev_dict, recall, precision = [], {}, {}, {}
    check_flag = True
    for _path in all_path_list:
        _basename = os.path.basename(_path)
        if _basename not in dump_basename_set:
            check_flag = False
            break
        _out_list = load_pickle(join(dump_dir, _basename))
        for _out in _out_list:
            accumulative_eval(_out["gold_answer"],_out["cur_question_type"], _out["prev_question_type"],_out["top1"],
                              _out["predicted_answer"],top1_pred, dev_dict, recall, precision)
    if check_flag:
        logging.info("!!! All data already exsiting in the dump dir {}".format(dump_dir))
        smp_result_print_wrt_qt(top1_pred, dev_dict, recall, precision)

    alter_ner_dir = join(infer_cfg.processed_dir, infer_cfg["ner_dump_dir"]) \
        if infer_cfg["use_dump_ner"] else None
    if infer_cfg["num_parallels"] == 1:
        eval_proc = EvalProcess(
            process_idxs[0], gpu_idxs[0], paths_list[0], infer_cfg["kb_mode"],
            dataset_obj, model_cfg, infer_cfg, dump_dir, infer_cfg["max_sequence_len"],
            verbose=infer_cfg["verbose_test"],
            use_filtered_ent=infer_cfg["use_filtered_ent"],
            alter_ner_dir=alter_ner_dir,
            use_op_type_constraint=infer_cfg["use_op_type_constraint"],
            timeout=infer_cfg["timeout"],
        )
        eval_proc.run()
    elif infer_cfg["num_parallels"] > 1:
        eval_procs = []
        for _i in range(infer_cfg["num_parallels"]):
            _proc = EvalProcess(
                process_idxs[_i], gpu_idxs[_i], paths_list[_i], infer_cfg["kb_mode"],
                dataset_obj, model_cfg, infer_cfg, dump_dir, infer_cfg["max_sequence_len"],
                verbose=infer_cfg["verbose_test"],
                use_filtered_ent=infer_cfg["use_filtered_ent"],
                alter_ner_dir=alter_ner_dir,
                use_op_type_constraint=infer_cfg["use_op_type_constraint"],
                timeout=infer_cfg["timeout"],
            )
            _proc.start()
            eval_procs.append(_proc)

        for _proc in eval_procs:
            _proc.join()
    else:
        raise AttributeError

    # aggregation
    logging.info("doing aggregation........")
    dump_path_list = get_data_path_list("all", dump_dir)
    dump_basename_list = [os.path.basename(_path) for _path in dump_path_list]
    dump_basename_set = set(dump_basename_list)
    top1_pred = []
    dev_dict = {}
    recall = {}
    precision = {}
    for _path in tqdm(all_path_list, total=len(all_path_list)):
        _basename = os.path.basename(_path)
        assert _basename in dump_basename_set
        _out_list = load_pickle(join(dump_dir, _basename))
        for _out in _out_list:
            accumulative_eval(
                _out["gold_answer"],
                _out["cur_question_type"],
                _out["prev_question_type"],
                _out["top1"], _out["predicted_answer"],
                top1_pred, dev_dict, recall, precision
            )
    smp_result_print_wrt_qt(top1_pred, dev_dict, recall, precision)


class EvalProcess(multiprocessing.Process):
    def __init__(
            self, idx, gpu_index, path_list, kb_mode,
            dataset_obj, model_cfg, infer_cfg, dump_dir,
            max_sequence_len,
            verbose=False,
            **kwargs  # use_filtered_ent, alter_ner_dir, use_op_type_constraint
    ):
        super(EvalProcess, self).__init__()
        self.idx = idx
        self.gpu_index = gpu_index
        self.path_list = path_list
        self.kb_mode = kb_mode
        self.dataset_obj = dataset_obj
        self.model_cfg = model_cfg
        self.infer_cfg = infer_cfg
        self.dump_dir = dump_dir
        self.max_sequence_len = max_sequence_len
        self.verbose = verbose
        assert os.path.exists(self.dump_dir) and os.path.isdir(self.dump_dir)

        # other in kwargs
        # # 1. use_filtered_ent
        use_filtered_ent = kwargs.get("use_filtered_ent")
        if isinstance(use_filtered_ent, bool) and use_filtered_ent:
            self.use_filtered_ent = True
        else:
            self.use_filtered_ent = False
        # # 2. alter_ner_dir
        alter_ner_dir = kwargs.get("alter_ner_dir")
        if isinstance(alter_ner_dir, str) and os.path.exists(alter_ner_dir) and os.path.isdir(alter_ner_dir):
            self.alter_ner_dir = alter_ner_dir
        else:
            self.alter_ner_dir = None
        # # 3. use_op_type_constraint
        use_op_type_constraint = kwargs.get("use_op_type_constraint")
        if isinstance(use_op_type_constraint, bool) and use_op_type_constraint:
            self.use_op_type_constraint = True
        else:
            self.use_op_type_constraint = False
        # # 4. timeout
        timeout = kwargs.get("timeout")
        if timeout is not None:
            self.timeout = timeout
        else:
            self.timeout = 5.

        logging.info("In process {}, use_filtered_ent is {}".format(self.idx, self.use_filtered_ent))
        logging.info("In process {}, alter_ner_dir is {}".format(self.idx, self.alter_ner_dir))
        logging.info("In process {}, use_op_type_constraint is {}".format(self.idx, self.use_op_type_constraint))

        self.daemon = True

    def run(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.infer_cfg['gpu'])
        feature_list = self.dataset_obj.process_test_data(self.path_list)
        g = tf.Graph()
        with g.as_default():
            with tf.device("/device:GPU:{}".format(self.gpu_index)):
                with tf.variable_scope('model') as scope:
                    # cfg, vocab, data_type, labels_dict, max_sequence_len, num_training_steps, scope
                    model_obj = self.model_cfg['model_class'](
                        self.model_cfg, self.dataset_obj.tokenizer, self.model_cfg['dataset'],
                        self.dataset_obj.get_labels_dict(), self.model_cfg["max_sequence_len"],
                        1000, scope.name
                    )
                graph_handler = GraphHandler(model_obj, self.infer_cfg)
                evaluator = E2eEvaluator(model_obj, self.infer_cfg)
                sess = graph_handler.initialize()
        # data preparation
        logging.info("loading inverse_index...")
        inverse_index = load_json("data/EDL/inverse_index_spacy_token.json") if self.alter_ner_dir is None else None
        logging.info("building lf executor")
        lf_executor = LfExecutor(kb_mode=self.kb_mode,use_op_type_constraint=self.use_op_type_constraint)
        logging.info("Done")

        # data in this process
        top1_pred = []
        dev_dict = {}
        recall = {}
        precision = {}
        _feature_ptr = 0
        for _idx_file, _file_path in tqdm(enumerate(self.path_list), total=len(self.path_list)):
            _dump_path = os.path.join(self.dump_dir, os.path.basename(_file_path))
            _raw_data = load_json(_file_path)
            assert len(_raw_data) % 2 == 0
            _num_turns = len(_raw_data) // 2
            # fetch the feature list
            _proc_features = feature_list[_feature_ptr: (_feature_ptr + _num_turns)]
            _feature_ptr += _num_turns
            # verity the equal of raw data and _proc_features
            for _idx_t in range(_num_turns):
                assert _raw_data[_idx_t * 2]["utterance"] == _proc_features[_idx_t]["utterances"]["cur_q"]
                assert _raw_data[_idx_t * 2 + 1]["utterance"] == _proc_features[_idx_t]["utterances"]["cur_a"]

            _out_list = None
            if os.path.exists(_dump_path) and os.path.isfile(_dump_path):
                try:
                    _out_list = load_pickle(_dump_path)
                    assert len(_out_list) == _num_turns
                    for _idx_t in range(_num_turns):
                        assert _out_list[_idx_t]["cur_question_type"] == _raw_data[_idx_t*2]["question-type"]
                except:
                    _out_list = None
            if _out_list is None:
                _out_list = evaluator.decoding(  # how to multi process
                    sess, _proc_features,
                    lf_executor, inverse_index, BaseProcessor.dict_e2t,
                    self.dataset_obj.get_labels_dict()["EOs"]["labels"],
                    self.dataset_obj.get_labels_dict()["sketch"]["labels"],
                    self.dataset_obj.get_labels_dict()["predicates"]["labels"],
                    self.dataset_obj.get_labels_dict()["types"]["labels"],
                    batch_size=20, max_seq_len=self.max_sequence_len,timeout=self.timeout,
                    use_filtered_ent=self.use_filtered_ent,
                    alter_ner_dir=self.alter_ner_dir,
                )
                assert len(_out_list) == _num_turns
                save_pickle(_out_list, _dump_path)

            if self.verbose:
                for _out in _out_list:
                    accumulative_eval(
                        _out["gold_answer"],
                        _out["cur_question_type"],
                        _out["prev_question_type"],
                        _out["top1"], _out["predicted_answer"],
                        top1_pred, dev_dict, recall, precision
                    )
            if self.verbose and (_idx_file+1) % 40 == 0:
                logging.info("")
                logging.info("="*30)
                logging.info("From process {}".format(self.idx))
                smp_result_print_wrt_qt(top1_pred, dev_dict, recall, precision)
                logging.info("="*30)

# ====== Utilities ======
def assign_gpu_idx(num_parallels, num_gpus):
    if isinstance(num_gpus, str):
        num_gpus = len(num_gpus.strip().split(","))
    idxs = list(range(num_parallels))
    gpu_idxs = [_i%num_gpus for _i in idxs]
    return idxs, gpu_idxs


