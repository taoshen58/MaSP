from e2e.e2e_dataset import E2eDataset, BaseProcessor
import tensorflow as tf
import numpy as np
import logging
import math
import os
import time
import collections
import csv
from tqdm import tqdm
from peach.utils.string import mp_join
from e2e.exe import LfExecutor
from copy import deepcopy
from utils.spacy_tk import spacy_tokenize
from utils.csqa import load_pickle, save_pickle

from peach.bert.utils import SPECIAL_TOKENS
UNK_TOKEN = SPECIAL_TOKENS["UNK"]
EMPTY_TOKEN = SPECIAL_TOKENS["EMPTY"]
SOS_TOKEN = SPECIAL_TOKENS["SOS"]
EOS_TOKEN = SPECIAL_TOKENS["EOS"]


class E2eEvaluator(object):
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

        self.run_dict = model.prediction_dict
        self.inverse_index = None
        self.dict_e2t = BaseProcessor.dict_e2t

    def get_evaluation(self, sess, dataset_obj, data_type, global_step=None, time_counter=None, *args, **kwargs):
        run_dict = self.run_dict

        assert isinstance(dataset_obj, E2eDataset)
        logging.info('getting evaluation result for %s' % data_type)
        data_name = dataset_obj.data_name
        example_batch_list = []

        if dataset_obj.data_name.startswith("e2e"):
            all_run_res = {
                "sep_indices": [],
                "EOs": [],
                "entity_types": [],
                "sketch": [],
                "sketch_entity": [],
                "sketch_predicate": [],
                "sketch_type": [],
                "sketch_num": [],
            }

            for example_batch, _, _, _ in dataset_obj.generate_batch_iter(
                    self.cfg['test_batch_size'], data_type):
                example_batch_list.extend(example_batch)
                feed_dict = self.model.get_feed_dict(example_batch, False)
                run_res = sess.run(run_dict, feed_dict=feed_dict)

                for _idx_ex, _example in enumerate(example_batch):
                    # ========= aux ============
                    sep_indices = run_res["sep_indices"][_idx_ex]
                    all_run_res["sep_indices"].append(list(sep_indices))  # len is 3

                    # ======== EOs and Entity type =========
                    seq_label_mask = run_res["seq_label_mask"][_idx_ex]
                    EOs = run_res["EOs"][_idx_ex]
                    entity_types = run_res["entity_types"][_idx_ex]

                    # transform from numpy type to list
                    seq_label_mask_bool = seq_label_mask.astype("bool")
                    EO_list = list(EOs[seq_label_mask_bool])
                    entity_type_list = list(entity_types[seq_label_mask_bool])
                    all_run_res["EOs"].append(EO_list)
                    all_run_res["entity_types"].append(entity_type_list)

                    # ========= sketch =======
                    sketch = run_res["sketch"][_idx_ex]
                    sketch_entity = run_res["sketch_entity"][_idx_ex]
                    sketch_predicate = run_res["sketch_predicate"][_idx_ex]
                    sketch_type = run_res["sketch_type"][_idx_ex]
                    sketch_num = run_res["sketch_num"][_idx_ex]

                    sketch_mask = sketch.astype("bool")
                    all_run_res["sketch"].append(list(sketch[sketch_mask]))
                    all_run_res["sketch_entity"].append(list(sketch_entity[sketch_mask]))
                    all_run_res["sketch_predicate"].append(list(sketch_predicate[sketch_mask]))
                    all_run_res["sketch_type"].append(list(sketch_type[sketch_mask]))
                    all_run_res["sketch_num"].append(list(sketch_num[sketch_mask]))

            # !!!!!!!!! dev: for EOs and entity type !!!!!!!!!!
            EO_f1_metric_obj = F1MetricV2()
            entity_type_f1_metric_obj = F1MetricV2()

            name_list = [
                "pa_EO",
                "pq_EO",
                "cq_EO",
                "pa_ET",
                "pq_ET",
                "cq_ET",
            ]
            f1_metric_obj_list = [
                F1MetricV2(),
                F1MetricV2(),
                F1MetricV2(),
                F1MetricV2(),
                F1MetricV2(),
                F1MetricV2(),
            ]
            for _idx_ex, _example in enumerate(example_batch_list):
                sep1, sep2, sep3 = all_run_res["sep_indices"][_idx_ex]
                EOs = all_run_res["EOs"][_idx_ex]
                entity_types = all_run_res["entity_types"][_idx_ex]

                assert len(EOs) == len(entity_types)
                assert len(EOs) == sep3

                gold_ids_list = [
                    _example["EOs_ids"]["prev_q"],
                    _example["EOs_ids"]["prev_a"],
                    _example["EOs_ids"]["cur_q"],
                    _example["entity_types_ids"]["prev_q"],
                    _example["entity_types_ids"]["prev_a"],
                    _example["entity_types_ids"]["cur_q"],
                ]
                pred_ids_list = [
                    EOs[:sep1],
                    EOs[(sep1 + 1):sep2],
                    EOs[(sep2 + 1):sep3],
                    entity_types[:sep1],
                    entity_types[(sep1 + 1):sep2],
                    entity_types[(sep2 + 1):sep3],
                ]

                for idx_o, (gold_ids, pred_ids, f1_metric_obj) in enumerate(zip(
                        gold_ids_list, pred_ids_list, f1_metric_obj_list
                )):
                    if len(gold_ids) == 0:
                        pred_ids = []
                    else:
                        gold_ids = gold_ids[:len(pred_ids)]
                    # remove the pad index for
                    gold_ids = [_elem - 1 for _elem in gold_ids]
                    pred_ids = [_elem - 1 for _elem in pred_ids]
                    f1_metric_obj.add_list(zip(gold_ids, pred_ids))

                    if idx_o < 3:
                        EO_f1_metric_obj.add_list(zip(gold_ids, pred_ids))
                    else:
                        entity_type_f1_metric_obj.add_list(zip(gold_ids, pred_ids))

            # aggregation
            all_result = {}
            EO_metric_dict = EO_f1_metric_obj.get_metric("EO")
            all_result.update(EO_metric_dict)
            entity_type_metric_dict = entity_type_f1_metric_obj.get_metric("ET")
            all_result.update(entity_type_metric_dict)
            # for name, f1_metric_obj in zip(name_list, f1_metric_obj_list):
            #     _metric = f1_metric_obj.get_metric(name)
            #     all_result.update(_metric)

            # !!!!!!!!!!!!! sketch !!!!!!!!!!!!!!!!!!
            sketch_counter = [0, 0]
            for _idx_e, (_example, _sketch) in enumerate(zip(
                    example_batch_list, all_run_res["sketch"])):
                if len(_example["lf"]["gold_sketch_ids"]) == 0:
                    continue
                _new_sketch = []
                for _id in _sketch:
                    if _id == self.model.sos_id:
                        pass
                    elif _id == self.model.eos_id:
                        pass
                    else:
                        _new_sketch.append(_id)
                if tuple(_new_sketch) == tuple(_example["lf"]["gold_sketch_ids"]):
                    sketch_counter[1] += 1
                else:
                    sketch_counter[0] += 1
            all_result["sketch_accuracy"] = 1. * sketch_counter[1] / sum(sketch_counter) \
                if sum(sketch_counter) > 0 else 0.

            # !!!!!!!!!!!!! leaves prediction !!!!!!!!!!!!!!!
            accu_counter_dict = {
                "sketch_entity": [0, 0],
                "sketch_predicate": [0, 0],
                "sketch_type": [0, 0],
                "sketch_num": [0, 0],
            }
            for _idx_ex, _example in enumerate(example_batch_list):
                gold_label_type = _example["lf"]["gold_sketch"]
                gold_label_list =_example["lf"]["gold_leaves_ids"]

                sketch_entity = all_run_res["sketch_entity"][_idx_ex]
                sketch_predicate = all_run_res["sketch_predicate"][_idx_ex]
                sketch_type = all_run_res["sketch_type"][_idx_ex]
                sketch_num = all_run_res["sketch_num"][_idx_ex]
                sep1, sep2, sep3 = all_run_res["sep_indices"][_idx_ex]
                base_len_dict = {
                    "prev_q": 0, "prev_a": sep1+1, "cur_q": sep2+1
                }

                if len(_example["lf"]["gold_sketch_ids"]) == 0:
                    continue

                for _idx_tk, (_skt_type, _skt_leaf) in enumerate(zip(gold_label_type, gold_label_list)):
                    if _idx_tk >= len(sketch_entity):
                        break
                    if _skt_type == "e":
                        gold_idx = base_len_dict[_skt_leaf[1]] + _skt_leaf[0]
                        if gold_idx == sketch_entity[_idx_tk]:
                            accu_counter_dict["sketch_entity"][1] += 1
                        else:
                            accu_counter_dict["sketch_entity"][0] += 1

                    elif _skt_type == "r":
                        if _skt_leaf == sketch_predicate[_idx_tk]:
                            accu_counter_dict["sketch_predicate"][1] += 1
                        else:
                            accu_counter_dict["sketch_predicate"][0] += 1
                    elif _skt_type == "Type":
                        if _skt_leaf == sketch_type[_idx_tk]:
                            accu_counter_dict["sketch_type"][1] += 1
                        else:
                            accu_counter_dict["sketch_type"][0] += 1
                    elif _skt_type == "num_utterence":
                        gold_idx = base_len_dict["cur_q"] + _skt_leaf
                        if gold_idx == sketch_num[_idx_tk]:
                            accu_counter_dict["sketch_num"][1] += 1
                        else:
                            accu_counter_dict["sketch_num"][0] += 1

            for key in accu_counter_dict:
                all_result[key+"_accuracy"] = 1. * accu_counter_dict[key][1] / sum(accu_counter_dict[key])\
                    if sum(accu_counter_dict[key]) > 0 else 0.

            # for key metric
            all_metric_strs = ["EO_f1_score", "ET_f1_score", "sketch_accuracy",
                               "sketch_entity_accuracy", "sketch_predicate_accuracy",
                               "sketch_type_accuracy", "sketch_num_accuracy"]
            all_weights = [0.5] * 2 + [1.] + [0.25] * 4
            assert len(all_metric_strs) == len(all_weights)
            all_result["key_metric"] = sum(all_result[_v]*_w for _v, _w in zip(all_metric_strs, all_weights)) \
                                       / sum(all_weights)
            # key_metric = 0.
            # c = 0
            # for key, val in all_result.items():
            #     if "f1" in key:
            #         key_metric += val
            #         c += 1
            # all_result["key_metric"] = key_metric / c if c > 0 else 0.

            return all_result

    def gen_ner_and_num_for_feature_list(
            self, sess, feature_list,
            ent_inverse_index, dict_e2t,
            EO_label_list, type_label_list,  # labels
            batch_size=48, max_seq_len=160,
            dump_dir=None
    ):
        valid_type_str_set = set(type_label_list[3:])

        prediction_dict = self.model.prediction_dict

        run_dict = {
            "EOs": prediction_dict["EOs"],
            "entity_types": prediction_dict["entity_types"],
            "seq_label_mask": prediction_dict["seq_label_mask"],
            "sep_indices": prediction_dict["sep_indices"],
        }

        example_list = feature_list
        ph_tk = EMPTY_TOKEN

        data_to_dump = {}  # from "Basename" to list of "needed data"
        for _idx_batch, _ex_ptr in tqdm(enumerate(range(0, len(example_list), batch_size)),
                                        total=math.ceil(1.*len(example_list)/batch_size)):
            _example_batch = example_list[_ex_ptr: (_ex_ptr + batch_size)]

            # feed dict and run
            feed_dict = self.model.get_feed_dict(_example_batch, False, max_seq_len=max_seq_len)
            run_res_np = sess.run(run_dict, feed_dict)
            for _idx_ex, _example in enumerate(_example_batch):
                _example_id = _example["id"].split("|||")
                assert len(_example_id) == 3
                _ex_basename, _ex_turn_idx, _ex_num_turns = _example_id[0], int(_example_id[1]), int(_example_id[2])
                if _ex_basename not in data_to_dump:
                    data_to_dump[_ex_basename] = [None for _ in range(_ex_num_turns)]

                # ========= aux ============
                sep_indices = run_res_np["sep_indices"][_idx_ex]
                sep1, sep2, sep3 = list(sep_indices)  # len is 3
                assert sep1 > 0 and sep2 - sep1 > 1 and sep3 - sep2 > 1  # there are empty placeholders for utterance

                # ======== EOs and Entity type =========
                seq_label_mask = run_res_np["seq_label_mask"][_idx_ex]
                EOs = run_res_np["EOs"][_idx_ex]
                entity_types = run_res_np["entity_types"][_idx_ex]

                # transform from numpy type to list
                seq_label_mask_bool = seq_label_mask.astype("bool")
                EO_id_list = list(EOs[seq_label_mask_bool])
                entity_type_id_list = list(entity_types[seq_label_mask_bool])

                # prepare concat tokenized
                _utterances = _example["tokenized_utterances"]
                prev_q_tk_list = _utterances["prev_q"].split()[:sep1]
                prev_q_tk_list = prev_q_tk_list if len(prev_q_tk_list) > 0 else [ph_tk]

                prev_a_tk_list = _utterances["prev_a"].split()[:(sep2 - sep1 - 1)]
                prev_a_tk_list = prev_a_tk_list if len(prev_a_tk_list) > 0 else [ph_tk]

                cur_q_tk_list = _utterances["cur_q"].split()[:(sep3 - sep2 - 1)]
                assert len(cur_q_tk_list) > 0
                all_token_list = prev_q_tk_list + [ph_tk] + prev_a_tk_list + [ph_tk] + cur_q_tk_list
                assert len(all_token_list) == len(EO_id_list) and len(all_token_list) == len(entity_type_id_list)

                # id2str for EO and entity type
                EO_list = [EO_label_list[_id] for _id in EO_id_list]
                entity_type_list = [type_label_list[_id] for _id in entity_type_id_list]

                ent_set_all_list = []
                extra_dict = {}
                ent_set_list, num_list = idx2entity_type_and_idx2_num_with(
                    all_token_list, EO_list, entity_type_list, valid_type_str_set, ph_tk,
                    ent_inverse_index, dict_e2t,
                    max_len=len(all_token_list) + 1,
                    cur_q_sep_idx=sep2,
                    ent_set_all_list_out=ent_set_all_list,
                    extra_dict_out=extra_dict,
                )
                extra_dict["seps"] = (sep1, sep2, sep3,)
                # formulate ent_set_all_list
                # assert len(ent_set_all_list) == len(ent_set_list)
                # for _idx_tk in range(len(ent_set_all_list)):
                #     if ent_set_list[_idx_tk] is None or ent_set_all_list[_idx_tk] is None:
                #         continue
                #     _slen = len(ent_set_list[_idx_tk])
                #     _llen = len(ent_set_all_list[_idx_tk])
                #     if _llen > _slen > 0:
                #         ent_set_all_list[_idx_tk] = set(list(ent_set_all_list[_idx_tk])[:_slen])  # incaseof Timeout

                # s
                pred_data = {
                    "max_seq_len": max_seq_len,
                    "entity_set_list": ent_set_list,
                    "entity_set_all_list": ent_set_all_list,
                    "num_list": num_list,
                }
                pred_data.update(extra_dict)
                data_to_dump[_ex_basename][_ex_turn_idx] = pred_data

        # check all the data
        # # no data can be None
        for _basename, _vals in data_to_dump.items():
            for _val in _vals:
                assert _val is not None, "{} is not complete".format(_basename)
        if isinstance(dump_dir, str):
            if not os.path.exists(dump_dir):
                os.mkdir(dump_dir)
            for _basename, _vals in data_to_dump.items():
                _dump_path = os.path.join(dump_dir, _basename)
                save_pickle(_vals, _dump_path)
        return data_to_dump

    def sketch_test(
            self, sess, feature_list,
            batch_size=48, max_seq_len=160,

    ):
        example_list = feature_list
        run_dict = self.run_dict
        all_run_res = {
            "sep_indices": [],
            "EOs": [],
            "entity_types": [],
            "sketch": [],
            "sketch_entity": [],
            "sketch_predicate": [],
            "sketch_type": [],
            "sketch_num": [],
        }

        for _idx_batch, _ex_ptr in tqdm(enumerate(range(0, len(example_list), batch_size)),
                                        total=math.ceil(1.*len(example_list)/batch_size)):
            _example_batch = example_list[_ex_ptr: (_ex_ptr+batch_size)]

            feed_dict = self.model.get_feed_dict(_example_batch, False, max_seq_len=max_seq_len)
            run_res = sess.run(run_dict, feed_dict=feed_dict)

            for _idx_ex, _example in enumerate(_example_batch):
                # ========= aux ============
                sep_indices = run_res["sep_indices"][_idx_ex]
                all_run_res["sep_indices"].append(list(sep_indices))  # len is 3

                # ======== EOs and Entity type =========
                seq_label_mask = run_res["seq_label_mask"][_idx_ex]
                EOs = run_res["EOs"][_idx_ex]
                entity_types = run_res["entity_types"][_idx_ex]

                # transform from numpy type to list
                seq_label_mask_bool = seq_label_mask.astype("bool")
                EO_list = list(EOs[seq_label_mask_bool])
                entity_type_list = list(entity_types[seq_label_mask_bool])
                all_run_res["EOs"].append(EO_list)
                all_run_res["entity_types"].append(entity_type_list)

                # ========= sketch =======
                sketch = run_res["sketch"][_idx_ex]
                sketch_entity = run_res["sketch_entity"][_idx_ex]
                sketch_predicate = run_res["sketch_predicate"][_idx_ex]
                sketch_type = run_res["sketch_type"][_idx_ex]
                sketch_num = run_res["sketch_num"][_idx_ex]

                sketch_mask = sketch.astype("bool")
                all_run_res["sketch"].append(list(sketch[sketch_mask]))
                all_run_res["sketch_entity"].append(list(sketch_entity[sketch_mask]))
                all_run_res["sketch_predicate"].append(list(sketch_predicate[sketch_mask]))
                all_run_res["sketch_type"].append(list(sketch_type[sketch_mask]))
                all_run_res["sketch_num"].append(list(sketch_num[sketch_mask]))
        # !!!!!!!!! dev: for EOs and entity type !!!!!!!!!!
        EO_f1_metric_obj = F1MetricV2()
        entity_type_f1_metric_obj = F1MetricV2()

        name_list = [
            "pa_EO", "pq_EO", "cq_EO", "pa_ET", "pq_ET", "cq_ET",
        ]
        f1_metric_obj_list = [
            F1MetricV2(), F1MetricV2(), F1MetricV2(), F1MetricV2(), F1MetricV2(), F1MetricV2(),
        ]
        for _idx_ex, _example in enumerate(example_list):
            sep1, sep2, sep3 = all_run_res["sep_indices"][_idx_ex]
            EOs = all_run_res["EOs"][_idx_ex]
            entity_types = all_run_res["entity_types"][_idx_ex]

            assert len(EOs) == len(entity_types)
            assert len(EOs) == sep3

            gold_ids_list = [
                _example["EOs_ids"]["prev_q"], _example["EOs_ids"]["prev_a"], _example["EOs_ids"]["cur_q"],
                _example["entity_types_ids"]["prev_q"], _example["entity_types_ids"]["prev_a"],
                _example["entity_types_ids"]["cur_q"],
            ]
            pred_ids_list = [
                EOs[:sep1],
                EOs[(sep1 + 1):sep2],
                EOs[(sep2 + 1):sep3],
                entity_types[:sep1],
                entity_types[(sep1 + 1):sep2],
                entity_types[(sep2 + 1):sep3],
            ]

            for idx_o, (gold_ids, pred_ids, f1_metric_obj) in enumerate(zip(
                    gold_ids_list, pred_ids_list, f1_metric_obj_list
            )):
                if len(gold_ids) == 0:
                    pred_ids = []
                else:
                    gold_ids = gold_ids[:len(pred_ids)]
                # remove the pad index for
                gold_ids = [_elem - 1 for _elem in gold_ids]
                pred_ids = [_elem - 1 for _elem in pred_ids]
                f1_metric_obj.add_list(zip(gold_ids, pred_ids))

                if idx_o < 3:
                    EO_f1_metric_obj.add_list(zip(gold_ids, pred_ids))
                else:
                    entity_type_f1_metric_obj.add_list(zip(gold_ids, pred_ids))

        # aggregation
        all_result = {}
        EO_metric_dict = EO_f1_metric_obj.get_metric("EO")
        all_result.update(EO_metric_dict)
        entity_type_metric_dict = entity_type_f1_metric_obj.get_metric("ET")
        all_result.update(entity_type_metric_dict)
        # for name, f1_metric_obj in zip(name_list, f1_metric_obj_list):
        #     _metric = f1_metric_obj.get_metric(name)
        #     all_result.update(_metric)

        # !!!!!!!!!!!!! sketch !!!!!!!!!!!!!!!!!!
        sketch_counter = [0, 0]
        sketch_counter_wrt_qt = {}
        for _idx_e, (_example, _sketch) in enumerate(zip(
                example_list, all_run_res["sketch"])):
            if len(_example["lf"]["gold_sketch_ids"]) == 0:
                continue
            if _example["question_type"] not in sketch_counter_wrt_qt:
                sketch_counter_wrt_qt[_example["question_type"]]= [0, 0]
            _new_sketch = []
            for _id in _sketch:
                if _id == self.model.sos_id:
                    pass
                elif _id == self.model.eos_id:
                    pass
                else:
                    _new_sketch.append(_id)
            if tuple(_new_sketch) == tuple(_example["lf"]["gold_sketch_ids"]):
                sketch_counter[1] += 1
                sketch_counter_wrt_qt[_example["question_type"]][1] += 1
            else:
                sketch_counter[0] += 1
                sketch_counter_wrt_qt[_example["question_type"]][0] += 1
        all_result["sketch_accuracy_all_qt"] = 1. * sketch_counter[1] / sum(sketch_counter) \
            if sum(sketch_counter) > 0 else 0.
        for _qt, _val in sketch_counter_wrt_qt.items():
            all_result["sketch_accuracy_{}".format(_qt)] = 1. * _val[1] / sum(_val) if sum(_val) > 0 else 0.

        # !!!!!!!!!!!!! leaves prediction !!!!!!!!!!!!!!!
        accu_counter_dict = {
            "sketch_entity": [0, 0],
            "sketch_predicate": [0, 0],
            "sketch_type": [0, 0],
            "sketch_num": [0, 0],
        }
        for _idx_ex, _example in enumerate(example_list):
            gold_label_type = _example["lf"]["gold_sketch"]
            gold_label_list = _example["lf"]["gold_leaves_ids"]

            sketch_entity = all_run_res["sketch_entity"][_idx_ex]
            sketch_predicate = all_run_res["sketch_predicate"][_idx_ex]
            sketch_type = all_run_res["sketch_type"][_idx_ex]
            sketch_num = all_run_res["sketch_num"][_idx_ex]
            sep1, sep2, sep3 = all_run_res["sep_indices"][_idx_ex]
            base_len_dict = {
                "prev_q": 0, "prev_a": sep1 + 1, "cur_q": sep2 + 1
            }

            if len(_example["lf"]["gold_sketch_ids"]) == 0:
                continue

            for _idx_tk, (_skt_type, _skt_leaf) in enumerate(zip(gold_label_type, gold_label_list)):
                if _idx_tk >= len(sketch_entity):
                    break
                if _skt_type == "e":
                    gold_idx = base_len_dict[_skt_leaf[1]] + _skt_leaf[0]
                    if gold_idx == sketch_entity[_idx_tk]:
                        accu_counter_dict["sketch_entity"][1] += 1
                    else:
                        accu_counter_dict["sketch_entity"][0] += 1

                elif _skt_type == "r":
                    if _skt_leaf == sketch_predicate[_idx_tk]:
                        accu_counter_dict["sketch_predicate"][1] += 1
                    else:
                        accu_counter_dict["sketch_predicate"][0] += 1
                elif _skt_type == "Type":
                    if _skt_leaf == sketch_type[_idx_tk]:
                        accu_counter_dict["sketch_type"][1] += 1
                    else:
                        accu_counter_dict["sketch_type"][0] += 1
                elif _skt_type == "num_utterence":
                    gold_idx = base_len_dict["cur_q"] + _skt_leaf
                    if gold_idx == sketch_num[_idx_tk]:
                        accu_counter_dict["sketch_num"][1] += 1
                    else:
                        accu_counter_dict["sketch_num"][0] += 1

        for key in accu_counter_dict:
            all_result[key + "_accuracy"] = 1. * accu_counter_dict[key][1] / sum(accu_counter_dict[key]) \
                if sum(accu_counter_dict[key]) > 0 else 0.
        return all_result

    def decoding(
            self, sess, feature_list, lf_executor,
            ent_inverse_index, dict_e2t,
            EO_label_list, sketch_label_list, predicate_label_list, type_label_list,  # labels
            batch_size=48, max_seq_len=160, timeout=5.,
            use_filtered_ent=True, alter_ner_dir=None,
            return_out_list=True, verbose=False,
    ):
        valid_type_str_set = set(type_label_list[3:])

        prediction_dict = self.model.prediction_dict
        decoder_dict = self.model.decoder_dict

        run_dict = {
            "EOs": prediction_dict["EOs"],
            "entity_types": prediction_dict["entity_types"],
            "seq_label_mask": prediction_dict["seq_label_mask"],
            "sep_indices": prediction_dict["sep_indices"],
            # states for decoding
            "encoder_states": decoder_dict["encoder_states_run"],
            "encoder_output_for_predicate": decoder_dict["encoder_output_for_predicate_run"],
            "encoder_output_for_type": decoder_dict["encoder_output_for_type_run"],
        }

        example_list = feature_list
        ph_tk = EMPTY_TOKEN

        top1_pred = []
        dev_dict = {}
        recall = {}
        precision = {}
        out_data_list = []
        for _idx_batch, _ex_ptr in enumerate(range(0, len(example_list), batch_size)):
            _example_batch = example_list[_ex_ptr: (_ex_ptr+batch_size)]

            # feed dict and run
            feed_dict = self.model.get_feed_dict(_example_batch, False, max_seq_len=max_seq_len)
            run_res_np = sess.run(run_dict, feed_dict)

            # entity and number in the utterance
            run_res = {
                "sep_indices": [],
                "EOs": [],
                "entity_types": [],
                "entity_set_all_list": [],
                "entity_set_list": [],
                "num_list": [],
            }

            for _idx_ex, _example in enumerate(_example_batch):
                if isinstance(alter_ner_dir, str):
                    _example_id = _example["id"].split("|||")
                    assert len(_example_id) == 3
                    _ex_basename, _ex_turn_idx, _ex_num_turns = _example_id[0], int(_example_id[1]), int(_example_id[2])
                    predicted_ner = load_pickle(os.path.join(alter_ner_dir, _ex_basename))
                    assert predicted_ner[_ex_turn_idx]["max_seq_len"] == max_seq_len
                    ent_set_list = predicted_ner[_ex_turn_idx]["entity_set_list"]
                    num_list = predicted_ner[_ex_turn_idx]["num_list"]
                    ent_set_all_list = predicted_ner[_ex_turn_idx]["entity_set_all_list"]
                else:
                    # ========= aux ============
                    sep_indices = run_res_np["sep_indices"][_idx_ex]
                    sep1, sep2, sep3 = list(sep_indices)  # len is 3
                    run_res["sep_indices"].append((sep1, sep2, sep3))
                    assert sep1 > 0 and sep2 - sep1 > 1 and sep3 - sep2 > 1 # have empty placeholders for utterance

                    # ======== EOs and Entity type =========
                    seq_label_mask = run_res_np["seq_label_mask"][_idx_ex]
                    EOs = run_res_np["EOs"][_idx_ex]
                    entity_types = run_res_np["entity_types"][_idx_ex]

                    # transform from numpy type to list
                    seq_label_mask_bool = seq_label_mask.astype("bool")
                    EO_id_list = list(EOs[seq_label_mask_bool])
                    entity_type_id_list = list(entity_types[seq_label_mask_bool])

                    # prepare concat tokenized
                    _utterances = _example["tokenized_utterances"]
                    prev_q_tk_list = _utterances["prev_q"].split()[:sep1]
                    prev_q_tk_list = prev_q_tk_list if len(prev_q_tk_list)>0 else [ph_tk]

                    prev_a_tk_list = _utterances["prev_a"].split()[:(sep2-sep1-1)]
                    prev_a_tk_list = prev_a_tk_list if len(prev_a_tk_list)>0 else [ph_tk]

                    cur_q_tk_list = _utterances["cur_q"].split()[:(sep3-sep2-1)]
                    assert len(cur_q_tk_list) > 0
                    all_token_list = prev_q_tk_list + [ph_tk] + prev_a_tk_list + [ph_tk] + cur_q_tk_list
                    assert len(all_token_list) == len(EO_id_list) and len(all_token_list) == len(entity_type_id_list)

                    # id2str for EO and entity type
                    EO_list = [EO_label_list[_id] for _id in EO_id_list]
                    entity_type_list = [type_label_list[_id] for _id in entity_type_id_list]

                    ent_set_all_list = []
                    ent_set_list, num_list = idx2entity_type_and_idx2_num_with(
                        all_token_list, EO_list, entity_type_list, valid_type_str_set, ph_tk,
                        ent_inverse_index, dict_e2t,
                        max_len=len(all_token_list)+1,
                        cur_q_sep_idx=sep2,
                        ent_set_all_list_out=ent_set_all_list)

                # formulate ent_set_all_list
                assert len(ent_set_all_list) == len(ent_set_list)
                for _idx_tk in range(len(ent_set_all_list)):
                    if ent_set_all_list[_idx_tk] is not None and len(ent_set_all_list[_idx_tk]) > 0:
                        ent_set_all_list[_idx_tk] = set(list(ent_set_all_list[_idx_tk])[:1])
                    if ent_set_list[_idx_tk] is not None and len(ent_set_list[_idx_tk]) > 0:
                        ent_set_list[_idx_tk] = set(list(ent_set_list[_idx_tk])[:1])
                    # if ent_set_list[_idx_tk] is None or ent_set_all_list[_idx_tk] is None:
                    #     continue
                    # _slen = len(ent_set_list[_idx_tk])
                    # _llen = len(ent_set_all_list[_idx_tk])
                    # if _llen > _slen > 0:
                    #     ent_set_all_list[_idx_tk] = set(list(ent_set_all_list[_idx_tk])[:_slen])  # incaseof Timeout

                run_res["entity_set_list"].append(ent_set_list)
                run_res["num_list"].append(num_list)
                run_res["entity_set_all_list"].append(ent_set_all_list)
                # print(all_token_list, EO_list, entity_type_list, ent_set_list, num_list)

            # run the decoder!
            dec_results = logical_form_decoder(
                sess, lf_executor, decoder_dict,
                run_res_np["encoder_states"],
                run_res_np["encoder_output_for_predicate"],
                run_res_np["encoder_output_for_type"],
                feed_dict[self.model.input_ids],
                self.model.get_init_decoder_history_np(len(_example_batch)),
                self.model.get_init_decoder_ids_np(len(_example_batch)),
                sketch_label_list, predicate_label_list, type_label_list,
                run_res["entity_set_list"] if use_filtered_ent else run_res["entity_set_all_list"],
                run_res["num_list"], beam_size=4, max_depth=35, max_accu_time=timeout
            )

            for _idx_ex in range(len(_example_batch)):
                _example = _example_batch[_idx_ex]
                _dec_result = dec_results[_idx_ex]

                gold_answer = answer_parser(
                    _example["all_entities"], _example["utterances"]["cur_a"], _example["question_type"])

                _candidate_results = []
                _candidate_lfs = []
                for (_type, _vals, _flag), _slf, _score in _dec_result:
                    assert _type == "S"
                    if not _flag:
                        _candidate_results.append(_vals)
                        _candidate_lfs.append(_slf)
                    else:
                        _candidate_results.extend(_vals)
                        _candidate_lfs.extend([_slf]*len(_vals))
                if len(_candidate_results) == 0:
                    pred_lf = []
                    pred_answer = []
                    top1 = False
                else:
                    pred_lf = _candidate_lfs[0]
                    pred_answer = _candidate_results[0]
                    if type(pred_answer) == int:
                        pred_answer = [pred_answer]
                    top1 = (pred_answer == gold_answer)

                if return_out_list:
                    out_data = {
                        "predicted_lf": pred_lf,
                        "predicted_answer": pred_answer,
                        "top1": top1,
                        # from gold
                        "gold_answer": gold_answer,
                        "cur_question_type": _example["question_type"],
                        "prev_question_type": _example["prev_question_type"],
                    }
                    out_data_list.append(out_data)

                if verbose:
                    logging.info("=" * 40)
                    logging.info("Batch progress: {}/{}".format(
                        _idx_batch, math.ceil(1.*len(example_list)/batch_size)))
                    accumulative_eval(
                        gold_answer, _example["question_type"], _example["prev_question_type"],
                        top1, pred_answer,
                        top1_pred, dev_dict, recall, precision
                    )
                    smp_result_print_wrt_qt(top1_pred, dev_dict, recall, precision)
        if verbose:
            logging.info("="*40)
            logging.info("="*40)
            smp_result_print_wrt_qt(top1_pred, dev_dict, recall, precision)
        return out_data_list


def accumulative_eval(
        answer, cur_qt, prev_qt,
        top1, pred_answer,
        top1_pred, dev_dict, recall, precision,
):
    top1_pred.append(top1)
    if cur_qt != 'Clarification':
        if cur_qt not in dev_dict:
            dev_dict[cur_qt] = [0.0, 0.0]
        if top1 is False:
            dev_dict[cur_qt][0] += 1
        else:
            dev_dict[cur_qt][1] += 1
        if type(answer) == set:
            if cur_qt not in recall:
                recall[cur_qt] = []
                precision[cur_qt] = []
            if type(pred_answer) == set:
                if len(answer) == 0 or len(pred_answer) == 0:
                    recall[cur_qt].append(0.0)
                    precision[cur_qt].append(0.0)
                else:
                    recall[cur_qt].append(len(answer & pred_answer) / len(answer))
                    precision[cur_qt].append(len(answer & pred_answer) / len(pred_answer))
            else:
                recall[cur_qt].append(0.0)
                precision[cur_qt].append(0.0)

        if prev_qt is not None and prev_qt == "Clarification":  # just for the Clarification
            if type(answer) == set:
                if prev_qt not in recall:
                    recall[prev_qt] = []
                    precision[prev_qt] = []
                if type(pred_answer) == set:
                    if len(answer) == 0 or len(pred_answer) == 0:
                        recall[prev_qt].append(0.0)
                        precision[prev_qt].append(0.0)
                    else:
                        recall[prev_qt].append(len(answer & pred_answer) / len(answer))
                        precision[prev_qt].append(len(answer & pred_answer) / len(pred_answer))
                else:
                    recall[prev_qt].append(0.0)
                    precision[prev_qt].append(0.0)


def smp_result_print_wrt_qt(top1_pred, dev_dict, recall, precision):
    logging.info("Number of examples: {}".format(len(top1_pred)))
    r=[]
    R={}
    for temp in recall:
        r+=recall[temp]
        R[temp]=np.mean(recall[temp])*100.0
    p=[]
    P={}
    for temp in precision:
        p+=precision[temp]
        P[temp]=np.mean(precision[temp])*100.0
    keys=sorted(list(set(dev_dict)|set(R)))
    logging.info("-"*100)
    logging.info("%-35s %-15s %-15s"%("","Recall","Precision"))
    logging.info("%-35s %-17.2f %-15.2f"%("Overall",np.mean(r)*100.0,np.mean(p)*100.0))
    for k in keys:
        if k in R:
            logging.info("%-35s %-17.2f %-15.2f"%(k,R[k],P[k]))
    logging.info("-"*100)
    logging.info("%-43s %-15s"%("","Accuracy"))
    for k in keys:
        if k not in R:
            logging.info("%-44s %-15.2f"%(k,dev_dict[k][1]*100.0/sum(dev_dict[k])))
    return (np.mean(r)+np.mean(p))/2.0*100.0


def answer_parser(all_entities, utterance, question_type):
    bool_answer = []
    cont_answer = []
    tokenized_utterance = spacy_tokenize(utterance).split()

    for token in tokenized_utterance:
        token = token.strip()
        if token in ["yes", "no"]:
            bool_answer.append(token)
        try:
            cont_answer.append(int(token))
        except ValueError:
            continue
    if "Bool" in question_type:
        return bool_answer
    elif "Count" in question_type:
        return cont_answer
    else:
        return set(all_entities)


def logical_form_decoder(
        sess, lf_executor, model_dec_dict,
        encoder_states_np_init,
        encoder_output_for_predicate_init, encoder_output_for_type_init,
        encoder_ids_np_init,
        decoder_history_np_init, decoder_ids_np_init,
        sketch_label_list, predicate_label_list, type_label_list,
        ent_set_lists, num_lists,
        beam_size, max_depth,
        max_accu_time=5.,
):
    no_repetition = False
    # only_first_entity = False

    isinstance(lf_executor, LfExecutor)
    # init decoder
    bs_init, esl, hn = encoder_states_np_init.shape
    run_dict = {
        "decoder_history": model_dec_dict["decoder_history_run"],
        "distribution_seq2seq": model_dec_dict["distribution_seq2seq_run"],
        "distribution_sketch_entity": model_dec_dict["distribution_sketch_entity_run"],
        "distribution_sketch_predicate": model_dec_dict["distribution_sketch_predicate_run"],
        "distribution_sketch_type": model_dec_dict["distribution_sketch_type_run"],
        "distribution_sketch_num": model_dec_dict["distribution_sketch_num_run"],
    }

    encoder_states_np = encoder_states_np_init
    encoder_output_for_predicate_np = encoder_output_for_predicate_init
    encoder_output_for_type_np = encoder_output_for_type_init
    encoder_ids_np = encoder_ids_np_init
    decoder_history_np = decoder_history_np_init
    decoder_ids_np = decoder_ids_np_init

    slfs_list = [[[], ] for _ in range(bs_init)]  # new
    answers_list = [[[], ] for _ in range(bs_init)]
    scores_list = [[0., ] for _ in range(bs_init)]
    valids_list = [[lf_executor.arg_set_lf_exe_with_next(_answer)[-1] for _answer in _answers]
                   for _answers in answers_list]
    time_record_list = [0. for _ in range(bs_init)]
    results_list = [[] for _ in range(bs_init)]  # list of (lf, answers, score)

    _cur_depth = 0
    while True:
        _cur_depth += 1

        feed_dict = {
            model_dec_dict["encoder_states_placeholder"]: encoder_states_np,
            model_dec_dict["encoder_output_for_predicate_placeholder"]: encoder_output_for_predicate_np,
            model_dec_dict["encoder_output_for_type_placeholder"]: encoder_output_for_type_np,  #
            model_dec_dict["encoder_ids_placeholder"]: encoder_ids_np,  #
            model_dec_dict["decoder_history_placeholder"]: decoder_history_np,  #
            model_dec_dict["decoder_ids_placeholder"]: decoder_ids_np,  #
            model_dec_dict["is_training_placeholder"]: False,  # fixed
        }
        run_res_dict = sess.run(run_dict, feed_dict=feed_dict)

        _row_ptr = 0
        branch_ptrs_list = [_i for _i in range(run_res_dict["decoder_history"].shape[0])]  # the ptr for run results
        new_branch_ptrs_list = []  # from new input to previous output for fetching the corresponding states
        for _idx_ex in range(bs_init):
            num_ex_tokens = len(ent_set_lists[_idx_ex])  # token list len for _idx_ex examples
            # index the intermediate results
            (_slfs, _answers, _scores, _valids) = \
                (slfs_list[_idx_ex], answers_list[_idx_ex], scores_list[_idx_ex], valids_list[_idx_ex])
            # sanity
            assert len(_slfs) == len(_answers) and len(_answers) == len(_scores)
            assert _row_ptr + len(_slfs) <= run_res_dict["distribution_seq2seq"].shape[0]

            # fetch run data
            _adists_seq2seq = run_res_dict["distribution_seq2seq"][_row_ptr:(_row_ptr + len(_slfs)), 0]
            _adists_sketch_entity = run_res_dict["distribution_sketch_entity"][_row_ptr:(_row_ptr + len(_slfs)), 0]
            _adists_sketch_predicate = run_res_dict["distribution_sketch_predicate"][_row_ptr:(_row_ptr + len(_slfs)), 0]
            _adists_sketch_type = run_res_dict["distribution_sketch_type"][_row_ptr:(_row_ptr + len(_slfs)), 0]
            _adists_sketch_num = run_res_dict["distribution_sketch_num"][_row_ptr:(_row_ptr + len(_slfs)), 0]
            _branch_ptrs = branch_ptrs_list[_row_ptr:(_row_ptr + len(_slfs))]
            _row_ptr += len(_slfs)

            _tmp_slfs = []
            _tmp_scores = []
            _tmp_answers = []
            _tmp_branch_ptrs = []

            for _idx_br in range(len(_slfs)):
                _slf, _answer, _score, _valid = _slfs[_idx_br], _answers[_idx_br], _scores[_idx_br], _valids[_idx_br]
                _dist = _adists_seq2seq[_idx_br]
                _dist_sketch_entity = _adists_sketch_entity[_idx_br][:num_ex_tokens]
                _dist_sketch_predicate = _adists_sketch_predicate[_idx_br]
                _dist_sketch_type = _adists_sketch_type[_idx_br]
                _dist_sketch_num = _adists_sketch_num[_idx_br][:num_ex_tokens]
                _branch_ptr = _branch_ptrs[_idx_br]

                # other constraints
                if no_repetition:
                    _exist_entities, _exist_predicates, _exist_types, _exist_nums = get_existing_leaves(_slf)
                else:
                    _exist_entities, _exist_predicates, _exist_types, _exist_nums = [], [], [], []

                # mask the distribution by the grammar
                assert len(_valid) > 0
                _valid_mask = [0.] * len(sketch_label_list)
                for _token in _valid:
                    _valid_mask[sketch_label_list.index(_token)] = 1.
                _valid_mask = np.array(_valid_mask, "float32")
                _masked_dist = _valid_mask * _dist
                deno = np.sum(_masked_dist)  # norm
                assert deno > 0.
                _masked_dist = _masked_dist / deno

                # fetch top k idxes and corresponding scores from _masked_dist
                _n_beam = min(beam_size, len(_valid))  # beam expansion on each branch

                _top_indices = np.argsort(-_masked_dist)[:_n_beam]
                _top_scores = np.log(_masked_dist[_top_indices])

                for _top_idx, _top_score in zip(_top_indices, _top_scores):
                    _new_token = sketch_label_list[_top_idx]
                    if _new_token == "e":  # sketch is "e", fetch the candidate entities
                        _n_beam_ent = min(beam_size, num_ex_tokens)
                        _ent_idxs, _ent_scores = sort_vec_with_mask(_dist_sketch_entity)
                        _candi_ent_sets = []
                        _candi_ent_scores = []
                        for _idx_i, (_ent_idx, _ent_score) in enumerate(zip(_ent_idxs, _ent_scores)):
                            _ent_set = ent_set_lists[_idx_ex][_ent_idx]
                            if _ent_set is None or len(_ent_set) == 0:
                                continue
                            if no_repetition and _ent_set in _exist_entities:
                                continue
                            if _ent_score == 0.:
                                break
                            if _ent_set not in _candi_ent_sets:
                                _candi_ent_sets.append(_ent_set)
                                _candi_ent_scores.append(_ent_score)  # only log the max score
                            if len(_candi_ent_sets) >= _n_beam_ent:
                                break
                            # if _idx_i >= _n_beam_ent:
                            #     break
                        if len(_candi_ent_sets) > 0:
                            # norm scores
                            _candi_ent_scores = score_list_norm(_candi_ent_scores)
                            for _ent_set, _ent_score in zip(_candi_ent_sets, _candi_ent_scores):
                                assert len(_ent_set) > 0
                                _flag = len(_ent_set) > 1
                                _ent_set = list(_ent_set) if _flag else list(_ent_set)[0]
                                _new_lf_elem = ("e", _ent_set, _flag)
                                _tmp_slfs.append(deepcopy(_slf) + [_new_lf_elem])
                                _tmp_answers.append(deepcopy(_answer) + [_new_lf_elem])
                                _tmp_scores.append(_score + _top_score + math.log(_ent_score + 1e-12))
                                _tmp_branch_ptrs.append(_branch_ptr)
                    elif _new_token == "r":
                        _n_beam_predicate = min(beam_size, np.sum(_dist_sketch_predicate > 0.))
                        _predicate_idxs, _predicate_scores = sort_vec_with_mask(_dist_sketch_predicate)
                        _candi_predicate_list = []
                        _candi_predicate_scores = []
                        for _p_idx, _p_score in zip(_predicate_idxs, _predicate_scores):
                            if _p_score == 0. or _p_idx < 3:  # not a special token
                                break
                            _predicate = predicate_label_list[_p_idx]
                            if no_repetition and _predicate in _exist_predicates:
                                continue
                            _candi_predicate_list.append(_predicate)
                            _candi_predicate_scores.append(_p_score)
                            if len(_candi_predicate_list) >= _n_beam_predicate:
                                break
                        _candi_predicate_scores = score_list_norm(_candi_predicate_scores)
                        for _p, _p_score in zip(_candi_predicate_list, _candi_predicate_scores):
                            _new_lf_elem = ("r", _p, False)
                            _tmp_slfs.append(deepcopy(_slf) + [_new_lf_elem])
                            _tmp_answers.append(deepcopy(_answer) + [_new_lf_elem])
                            _tmp_scores.append(_score + _top_score + math.log(_p_score + 1e-12))
                            _tmp_branch_ptrs.append(_branch_ptr)
                    elif _new_token == "Type":
                        _n_beam_type = min(beam_size, np.sum(_dist_sketch_type > 0.))
                        _type_idxs, _type_scores = sort_vec_with_mask(_dist_sketch_type)
                        _candi_type_list = []
                        _candi_type_scores = []
                        for _t_idx, _t_score in zip(_type_idxs, _type_scores):
                            if _t_score == 0. or _t_idx < 3:  # not a special token
                                break
                            _type = type_label_list[_t_idx]
                            if no_repetition and _type in _exist_types:
                                continue
                            _candi_type_list.append(_type)
                            _candi_type_scores.append(_t_score)
                            if len(_candi_type_list) >= _n_beam_type:
                                break
                        _candi_type_scores = score_list_norm(_candi_type_scores)
                        for _t, _t_score in zip(_candi_type_list, _candi_type_scores):
                            _new_lf_elem = ("Type", _t, False)
                            _tmp_slfs.append(deepcopy(_slf) + [_new_lf_elem])
                            _tmp_answers.append(deepcopy(_answer) + [_new_lf_elem])
                            _tmp_scores.append(_score + _top_score + math.log(_t_score + 1e-12))
                            _tmp_branch_ptrs.append(_branch_ptr)
                    elif _new_token == "num_utterence":
                        _n_beam_num = min(beam_size, num_ex_tokens)
                        _num_idxs, _num_scores = sort_vec_with_mask(_dist_sketch_num)

                        _candi_num_list = []
                        _candi_num_scores = []
                        for _num_idx, _num_score in zip(_num_idxs, _num_scores):
                            _num = num_lists[_idx_ex][_num_idx]
                            if _num is None:
                                continue
                            if no_repetition and _num in _exist_nums:
                                continue
                            if _num_score == 0.:
                                break
                            if _num not in _candi_num_list:
                                _candi_num_list.append(_num)
                                _candi_num_scores.append(_num_score)
                            if len(_candi_num_list) > _n_beam_num:
                                break
                        if len(_candi_num_list) > 0:
                            _candi_num_scores = score_list_norm(_candi_num_scores)
                            for _num, _num_score in zip(_candi_num_list, _candi_num_scores):
                                _new_lf_elem = ("num_utterence", _num, False)
                                _tmp_slfs.append(deepcopy(_slf) + [_new_lf_elem])
                                _tmp_answers.append(deepcopy(_answer) + [_new_lf_elem])
                                _tmp_scores.append(_score + _top_score + math.log(_num_score + 1e-12))
                                _tmp_branch_ptrs.append(_branch_ptr)
                    else:
                        assert _new_token.startswith("A")
                        _new_lf_elem = ("Action", _new_token, False)
                        _tmp_slfs.append(deepcopy(_slf) + [_new_lf_elem])
                        _tmp_answers.append(deepcopy(_answer) + [_new_lf_elem])
                        _tmp_scores.append(_score+_top_score)
                        _tmp_branch_ptrs.append(_branch_ptr)
            # get top beam
            _idxs_to_keep = list(np.argsort(-np.array(_tmp_scores)))
            _new_slfs = []
            _new_scores = []
            _new_answers = []
            _new_valids = []
            _new_branch_ptrs = []
            _timeout_flag = False
            for _idx in _idxs_to_keep:  # the idx is top idx
                time_s = time.time()
                _is_succ, _run_res, next_val = lf_executor.arg_set_lf_exe_with_next(_tmp_answers[_idx])
                time_record_list[_idx_ex] += time.time() - time_s
                if _is_succ == False:
                    continue
                elif _is_succ == True:
                    assert len(_run_res) == 1
                    results_list[_idx_ex].append((_run_res[0], _tmp_slfs[_idx], _tmp_scores[_idx]))
                else:  # _is_succ is None
                    _new_slfs.append(_tmp_slfs[_idx])
                    _new_scores.append(_tmp_scores[_idx])
                    _new_answers.append(_run_res)
                    _new_valids.append(next_val)
                    _new_branch_ptrs.append(_tmp_branch_ptrs[_idx])
                if len(_new_slfs) >= beam_size:
                    break
                if max_accu_time is not None and time_record_list[_idx_ex] > max_accu_time:
                    _timeout_flag = True
                    break
            # update the list in the main tracks or stop
            if len(results_list[_idx_ex]) < beam_size and _cur_depth < max_depth and not _timeout_flag:
                slfs_list[_idx_ex] = _new_slfs
                scores_list[_idx_ex] = _new_scores
                answers_list[_idx_ex] = _new_answers
                valids_list[_idx_ex] = _new_valids
                new_branch_ptrs_list.extend(_new_branch_ptrs)
            else:
                slfs_list[_idx_ex], scores_list[_idx_ex], answers_list[_idx_ex], valids_list[_idx_ex] = \
                    [], [], [], []
                new_branch_ptrs_list.extend([])
        # !!! end of example for iteration
        # construct next decoder inputs
        decoder_inputs = []
        for _idx_ex in range(bs_init):
            for _candi_sketch in slfs_list[_idx_ex]:
                _lf_elem = _candi_sketch[-1]
                _dec_inp_tk = _lf_elem[1] if _lf_elem[0] == "Action" else _lf_elem[0]
                decoder_inputs.append(sketch_label_list.index(_dec_inp_tk))
        assert len(decoder_inputs) == len(new_branch_ptrs_list)
        if len(decoder_inputs) > 0:
            # 1. encoder_states: encoder_states_np
            encoder_states_np = np.stack([encoder_states_np[_ptr] for _ptr in new_branch_ptrs_list], axis=0)
            # 1.1 encoder_output_for_predicate_np encoder_output_for_type_np
            encoder_output_for_predicate_np = np.stack(
                [encoder_output_for_predicate_np[_ptr] for _ptr in new_branch_ptrs_list], axis=0)
            encoder_output_for_type_np = np.stack(
                [encoder_output_for_type_np[_ptr] for _ptr in new_branch_ptrs_list], axis=0)
            # 2. encoder_ids:  encoder_ids_np
            encoder_ids_np = np.stack([encoder_ids_np[_ptr] for _ptr in new_branch_ptrs_list], axis=0)
            # 3. decoder_history: decoder_history_np
            decoder_history_np = np.stack([run_res_dict["decoder_history"][_ptr] for
                                           _ptr in new_branch_ptrs_list], axis=0)
            # 4. decoder_ids: decoder_ids_np
            decoder_ids_np = np.array(decoder_inputs, "int32").reshape([len(decoder_inputs), 1])
        else:  # stop the beam search
            break
    # !!! end of beam search
    for _idx_ex in range(bs_init):
        results_list[_idx_ex] = list(sorted(results_list[_idx_ex], key=lambda elem: elem[-1], reverse=True))
    return results_list  # list (ex) of list (candidate) of tuple(Answer,Set_arg LF, score)


def get_existing_leaves(lf):
    ents = []
    predicates = []
    types = []
    nums = []
    for _idx in range(len(lf)):
        _elem = lf[_idx]
        _type = _elem[0]
        _val = _elem[1]
        if _type == "e":
            ents.append(_val)
        elif _type == "r":
            predicates.append(_val)
        elif _type == "Type":
            types.append(_val)
        elif _type == "num_utterence":
            nums.append(_val)
    return ents, predicates, types, nums


def score_list_norm(scores):
    if len(scores) == 0:
        return scores
    for _s in scores:
        assert _s >=0
    ssum = sum(scores)
    if ssum > 0:
        return [_s/ssum for _s in scores]
    else:
        return [1./len(scores) for _ in scores]


def sort_vec_with_mask(vec_inp, mask=None, length=None):
    if mask is not None:
        vec_inp = vec_inp * mask.astype("float32")
    _top_idxs = np.argsort(-vec_inp)
    if length is not None:
        _top_idxs = _top_idxs[:length]
    _top_scores = vec_inp[_top_idxs]
    return _top_idxs, _top_scores


def idx2entity_type_and_idx2_num_with(
        token_list, EO_list, entity_type_list, valid_type_str_set,
        placeholder_token, inverse_index, dict_e2t, max_len=None, cur_q_sep_idx=None,
        ent_set_all_list_out=None, extra_dict_out=None
):
    max_len = max_len or len(token_list)
    def _dict_idx2any_to_list(_dict_idx2any, max_len):
        _exist_idxs = list(sorted(_dict_idx2any.keys()))
        if len(_exist_idxs) == 0:
            return [None] * max_len

        _res_list = []
        for _idx_a in range(max_len):
            if _idx_a in _dict_idx2any:
                _res_list.append(_dict_idx2any[_idx_a])
            else:
                __min_val = max_len + 1
                __min_idx = None
                for __idx_et in range(len(_exist_idxs)):
                    __dist = _idx_a - _exist_idxs[__idx_et]
                    __dist = __dist if __dist >= 0 else -__dist
                    if __dist <= __min_val:
                        __min_val = __dist
                        __min_idx = __idx_et

                assert __min_idx is not None
                _res_list.append(_dict_idx2any[_exist_idxs[__min_idx]])
        return _res_list

    # process the entity
    pred_entity_candidates, pred_entity_types, pred_entity_indices = \
        ner_with_type_and_indices(token_list, EO_list, entity_type_list, placeholder_token)

    dict_idx2ents = {}
    dict_idx2ents_all = {}
    for _idx_ent in range(len(pred_entity_candidates)):
        _ent_str = pred_entity_candidates[_idx_ent]
        _ent_type = pred_entity_types[_idx_ent]
        _ent_idxs = pred_entity_indices[_idx_ent]

        try:
            ent_list_all = inverse_index[_ent_str]['idxs']
            if _ent_type in valid_type_str_set:
                ent_list = [_ent for _ent in ent_list_all if _ent in dict_e2t and dict_e2t[_ent] == _ent_type]
                if len(ent_list) == 0:
                    ent_list = ent_list_all
            else:
                ent_list = ent_list_all

            for _idx_tk in _ent_idxs:
                dict_idx2ents_all[_idx_tk] = set(ent_list_all)
                dict_idx2ents[_idx_tk] = set(ent_list)
        except KeyError:
            pass
    ent_set_list = _dict_idx2any_to_list(dict_idx2ents, max_len)
    if isinstance(ent_set_all_list_out, list):
        assert len(ent_set_all_list_out) == 0
        ent_set_all_list = _dict_idx2any_to_list(dict_idx2ents_all, max_len)
        ent_set_all_list_out.extend(ent_set_all_list)

    # process the num
    dict_idx2num = {}
    for _idx_tk in range(len(token_list)):
        if _idx_tk not in dict_idx2ents:
            try:
                _num = int(token_list[_idx_tk])
                dict_idx2num[_idx_tk] = _num
            except ValueError:
                pass
    if cur_q_sep_idx is not None:
        for _idx in list(dict_idx2num.keys()):
            if _idx < int(cur_q_sep_idx):
                dict_idx2num.pop(_idx)
    num_list = _dict_idx2any_to_list(dict_idx2num, max_len)
    if isinstance(extra_dict_out, dict):
        extra_dict_out["dict_idx2ents"] = dict_idx2ents
        extra_dict_out["dict_idx2ents_all"] = dict_idx2ents_all
        extra_dict_out["dict_idx2num"] = dict_idx2num
    return ent_set_list, num_list


def ner_with_type_and_indices(token_list, EO_list, entity_type_list, placeholder_token=None):
    def _get_et(et_list):
        most_comm = collections.Counter(et_list).most_common()
        if len(most_comm) == 0:
            return UNK_TOKEN
        else:
            return most_comm[0][0]

    assert len(token_list) == len(EO_list) and len(EO_list) == len(entity_type_list)
    pred_entity_candidates = []
    pred_entity_types = []
    pred_entity_indices = []

    e = []
    t = []
    idxs = []
    flag = False  # not in a entity
    for _idx_t in range(len(token_list)):
        EO_val = EO_list[_idx_t]
        if placeholder_token is not None and token_list[_idx_t] == placeholder_token:
            EO_val = "SEP"

        if EO_val == "B":  # OK
            if flag:
                pred_entity_candidates.append(" ".join(e))
                pred_entity_types.append(_get_et(t))
                pred_entity_indices.append(idxs)
                e, t, idxs = [], [], []
            e.append(token_list[_idx_t])
            t.append(entity_type_list[_idx_t])
            idxs.append(_idx_t)
            flag = True
        elif EO_val == "S":  # OK
            if flag:
                pred_entity_candidates.append(" ".join(e))
                pred_entity_types.append(_get_et(t))
                pred_entity_indices.append(idxs)
            e, t, idxs = [], [], []
            flag = False
            pred_entity_candidates.append(token_list[_idx_t])
            pred_entity_types.append(entity_type_list[_idx_t])
            pred_entity_indices.append([_idx_t])
        elif EO_val == "M":  # OK
            if not flag:
                flag = True
            e.append(token_list[_idx_t])
            t.append(entity_type_list[_idx_t])
            idxs.append(_idx_t)

        elif EO_val in ["O", "SEP"]:  # OK
            if flag:
                # if EO_val == "O":
                #     e.append(token_list[_idx_t])
                #     t.append(entity_type_list[_idx_t])
                #     idxs.append(_idx_t)
                pred_entity_candidates.append(" ".join(e))
                pred_entity_types.append(_get_et(t))
                pred_entity_indices.append(idxs)
            e, t, idxs = [], [], []
            flag = False
        elif EO_val == "E":  # OK
            if flag:
                e.append(token_list[_idx_t])
                t.append(entity_type_list[_idx_t])
                idxs.append(_idx_t)
                pred_entity_candidates.append(" ".join(e))
                pred_entity_types.append(_get_et(t))
                pred_entity_indices.append(idxs)
            else:
                pred_entity_candidates.append(token_list[_idx_t])
                pred_entity_types.append(entity_type_list[_idx_t])
                pred_entity_indices.append([_idx_t])
            e, t, idxs = [], [], []
            flag = False
        else:
            assert AttributeError

        if len(e) > 0:
            assert len(e) == len(t) and len(e) == len(idxs)
            assert flag

    assert len(pred_entity_candidates) == len(pred_entity_types) \
           and len(pred_entity_candidates) == len(pred_entity_indices)
    return pred_entity_candidates, pred_entity_types, pred_entity_indices


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
