from os.path import join
import peach.bert.tokenization as tokenization
import tensorflow as tf
import numpy as np
import logging
import os
import csv
import math
import random
import copy
from peach.utils.data.dataset import data_queue, data_traverse
from peach.utils.data.seq import get_transition_params_from_sequence_list
from peach.utils.string import matching_by_voting
import glob
import os
import json
from tqdm import tqdm
from functools import reduce
import logging
import collections
from utils.csqa import load_json, transform_turn_list_to_history_format, \
    get_data_path_list, get_utterance, get_tokenized_utterance, get_entities, get_predicates, get_types,\
    get_EOs, get_entity_types, generate_EO_with_etype, index_num_in_tokenized_utterance
import spacy
from utils.spacy_tk import spacy_tokenize
from peach.bert.utils import SPECIAL_TOKENS, SPECIAL_TOKEN_MAPPING
PAD_TOKEN = SPECIAL_TOKENS["PAD"]
UNK_TOKEN = SPECIAL_TOKENS["UNK"]
EMPTY_TOKEN = SPECIAL_TOKENS["EMPTY"]
SOS_TOKEN = SPECIAL_TOKENS["SOS"]
EOS_TOKEN = SPECIAL_TOKENS["EOS"]


class E2eDataset(object):
    def __init__(
            self, data_name, data_dirs, vocab_file, do_lower_case,
            labels_dict_for_infer=None
    ):
        self.data_types = ["train", "dev"]
        assert len(data_dirs) == 2
        self.data_name = data_name
        self.train_data_dir, self.dev_data_dir = data_dirs
        self.vocab_file = vocab_file
        self.do_lower_case = do_lower_case

        self.processor = E2eProcessor()

        # 1. tokenize
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)

        # placeholders
        self._train_feature_list, self._dev_feature_list = [], []
        self._labels_dict = labels_dict_for_infer or {}

    def clean_data(self, del_lfs=True):
        for _example in self._train_feature_list:
            if del_lfs:
                if "lf" in _example and len(_example["lf"]["true_lf"]) > 0:
                        if len(_example["lf"]["valid_indices"]) == 0:
                            _example["lf"]["true_lf"] = _example["lf"]["true_lf"][:1]
                        else:
                            _example["lf"]["valid_indices"] = _example["lf"]["valid_indices"][:2]
                            _example["lf"]["true_lf"] = [
                                _example["lf"]["true_lf"][_idx]
                                for _idx in _example["lf"]["valid_indices"]]


    def generate_batch_iter(self, batch_size, data_type, max_step=None, *args, **kwargs):
        assert data_type in ['train', 'dev', ]
        nn_data = getattr(self, '_{}_feature_list'.format(data_type))
        batch_num = math.ceil(len(nn_data) / batch_size)
        data_out_fn = data_queue if isinstance(max_step, int) else data_traverse
        for sample_batch, data_round, idx_b in data_out_fn(nn_data, batch_size, max_step, *args, **kwargs):
            yield sample_batch, batch_num, data_round, idx_b

    # add test data processing
    def process_test_data(self, data_dir):
        logging.info("For testing data")
        example_list = self._pre_process_raw_data(data_dir)
        example_list, = self.processor.get_updated_example_list([example_list], update_labels=False)
        logging.info("\ttransforming examples to features")
        feature_list = []
        for _example in tqdm(example_list):
            feature_list.append(
                self.processor.transform_example_to_feature(_example, self.processor.get_labels_dict(), self.tokenizer),
            )
        return feature_list

    def fetch_examples(self, max_sent_len=None):

        data_dict_wrt_qt = {}
        for _example in tqdm(self._train_feature_list):
            if _example["question_type"] not in data_dict_wrt_qt:
                data_dict_wrt_qt[_example["question_type"]] = []
            # filter
            if "lf" not in _example or _example["lf"]["gold_sketch"] is None:
                continue
            if len(_example["utterances"]["prev_q"]) == 0 or \
                    _example["utterances"]["prev_q"].lower() in ["yes", "no"]\
                    or _example["utterances"]["prev_q"].lower().strip().startswith("and "):
                continue

            if _example["prev_question_type"] in ["Verification (Boolean) (All)", "Clarification"]:
                continue
            sent_len = sum(len(_example["tokenized_utterances"][_ut].split()) for _ut in ["cur_q", "prev_a", "prev_q"])
            if max_sent_len is not None and sent_len > max_sent_len:
                continue
            data_dict_wrt_qt[_example["question_type"]].append(_example)
        # shuffle
        random.seed(2020)
        for _qt in data_dict_wrt_qt:
            random.shuffle(data_dict_wrt_qt[_qt])
        # dataset_obj.fetch_examples(max_sent_len=13)
        anchor = 0

    def process_training_data(self, debug_num=0):
        # train
        logging.info("For training data")
        train_example_list = self._pre_process_raw_data(self.train_data_dir, debug_num=debug_num)
        train_example_list, = self.processor.get_updated_example_list([train_example_list], update_labels=True)
        logging.info("\ttransforming examples to features")
        self._train_feature_list = []
        for _example in tqdm(train_example_list):
            self._train_feature_list.append(
                self.processor.transform_example_to_feature(_example, self.processor.get_labels_dict(), self.tokenizer),
            )
        self._labels_dict = self.processor.get_labels_dict()

        # dev
        logging.info("For dev data")
        dev_example_list = self._pre_process_raw_data(self.dev_data_dir, debug_num=debug_num // 10)
        dev_example_list, = self.processor.get_updated_example_list([dev_example_list], update_labels=False)
        logging.info("\ttransforming examples to features")
        self._dev_feature_list = []
        for _example in tqdm(dev_example_list):
            self._dev_feature_list.append(
                self.processor.transform_example_to_feature(_example, self.processor.get_labels_dict(), self.tokenizer),
            )

    @property
    def num_train_examples(self):
        return len(getattr(self, "_train_feature_list"))

    def get_labels_dict(self):
        return self._labels_dict

    def _pre_process_raw_data(self, path_list, debug_num=0):
        # 1. data
        if isinstance(path_list, list):
            path_list = path_list
        else:
            path_list = get_data_path_list("all", path_list)

        # for debug:
        if debug_num is not None and debug_num > 0:
            path_list = path_list[:debug_num]

        # formulate the data
        logging.info("\tFormulating the raw data data")
        turn_list = []
        for idx_f, file_path in tqdm(enumerate(path_list)):
            raw_data = load_json(file_path)
            new_turn_list = self._get_formulated_dialog(raw_data, file_path)
            # some other processes
            turn_list.extend(new_turn_list)
        return turn_list

    @staticmethod
    def _get_formulated_dialog(raw_data, file_path):
        def _get_ent2idxss(_dialog_elem):
            if _dialog_elem is None:
                return {}
            return _dialog_elem.get("ent2idxss") or {}

        for _d in raw_data:
            _d["tokenized_utterance"] = spacy_tokenize(_d["utterance"])
            temp = get_entities(_d).copy()
            temp_ent_types = [BaseProcessor.dict_e2t[ent] if ent in BaseProcessor.dict_e2t else UNK_TOKEN
                              for ent in temp]
            temp_strs = [BaseProcessor.dict_e[idx] for idx in temp]

            if len(temp) > 0:
                temp, temp_ent_types, temp_strs = zip(
                    *list(
                        sorted(zip(temp, temp_ent_types, temp_strs),
                               key=lambda elem: len(elem[2].split()), reverse=True)
                    ))
            EO, entity_types, dict_code2idxss = generate_EO_with_etype(
                _d["tokenized_utterance"], temp, temp_strs, temp_ent_types, EMPTY_TOKEN)
            assert len(_d["tokenized_utterance"].split()) == len(EO)
            _d["EOs"] = EO
            _d["entity_types"] = entity_types
            _d["ent2idxss"] = dict_code2idxss

        new_dialog = []
        for idx_turn, raw_dialog_turn in enumerate(transform_turn_list_to_history_format(raw_data)):
            prev_none = raw_dialog_turn["previous"] is None
            prev_q = None if prev_none else raw_dialog_turn["previous"]["user"]
            prev_a = None if prev_none else raw_dialog_turn["previous"]["system"]
            cur_q = raw_dialog_turn["current"]["user"]
            cur_a = raw_dialog_turn["current"]["system"]

            turn = {
                "id": "{}|||{}|||{}".format(os.path.basename(file_path), idx_turn, len(raw_data)//2),
                "question_type": raw_dialog_turn["question_type"],
                "prev_question_type": prev_q["question-type"] if prev_q is not None else None,
                "description": cur_q.get("description"),
                "utterances": {
                    "prev_q": get_utterance(prev_q),
                    "prev_a": get_utterance(prev_a),
                    "cur_q": get_utterance(cur_q),
                    "cur_a": get_utterance(cur_a),
                },
                "tokenized_utterances": {
                    "prev_q": get_tokenized_utterance(prev_q),
                    "prev_a": get_tokenized_utterance(prev_a),
                    "cur_q": get_tokenized_utterance(cur_q),
                    "cur_a": get_tokenized_utterance(cur_a),
                },
                "entities": {
                    "prev_q": get_entities(prev_q),
                    "prev_a": get_entities(prev_a),
                    "cur_q": get_entities(cur_q),
                    "cur_a": get_entities(cur_a),
                },
                "predicates": {
                    "prev_q": get_predicates(prev_q),
                    "cur_q": get_predicates(cur_q),
                },
                "types": {
                    "prev_q": get_types(prev_q),
                    "cur_q": get_types(cur_q),
                },
                "EOs": {
                    "prev_q": get_EOs(prev_q),
                    "prev_a": get_EOs(prev_a),
                    "cur_q": get_EOs(cur_q),
                    "cur_a": get_EOs(cur_a),
                },
                "entity_types": {
                    "prev_q": get_entity_types(prev_q),
                    "prev_a": get_entity_types(prev_a),
                    "cur_q": get_entity_types(cur_q),
                    "cur_a": get_entity_types(cur_a),
                },
                "ent2idxss":{
                    "prev_q": _get_ent2idxss(prev_q),
                    "prev_a": _get_ent2idxss(prev_a),
                    "cur_q": _get_ent2idxss(cur_q),
                    "cur_a": _get_ent2idxss(cur_a),
                },
            }
            if "all_entities" in raw_dialog_turn["current"]["system"]:
                turn["all_entities"] = raw_dialog_turn["current"]["system"]["all_entities"]
            else:
                assert AttributeError

            if "true_lf" in raw_dialog_turn["current"]["system"]:
                turn["lf"] = {
                    "true_lf": raw_dialog_turn["current"]["system"]["true_lf"],
                }
            new_dialog.append(turn)
        return new_dialog

    # ============ log utilities =========
    def log_lf_success_ratio_after_filtering(self):
        logging.info("log lf success ratio after filtering")

        dict_qt2accu_src = {}
        dict_qt2accu_tgt = {}

        for _example in self._train_feature_list:
            _qt = _example["question_type"]
            # before filtering
            if _qt not in dict_qt2accu_src:
                dict_qt2accu_src[_qt] = [0, 0]
            if len(_example["lf"]["true_lf"]) > 0:
                dict_qt2accu_src[_qt][1] += 1
            else:
                dict_qt2accu_src[_qt][0] += 1
            # after filtering
            if _qt not in dict_qt2accu_tgt:
                dict_qt2accu_tgt[_qt] = [0, 0]
            if _example["lf"]["gold_sketch"] is not None and len(_example["lf"]["gold_sketch_ids"]) > 0:
                dict_qt2accu_tgt[_qt][1] += 1
            else:
                dict_qt2accu_tgt[_qt][0] += 1

        assert len(dict_qt2accu_src) == len(dict_qt2accu_tgt)

        for _qt in dict_qt2accu_tgt:
            _ct_src = dict_qt2accu_src[_qt]
            _ct_tgt = dict_qt2accu_tgt[_qt]
            logging.info("\tfor {}, num is {}, success ratio:".format(
                _qt, sum(_ct_src)))
            print("\t\t before filtering {}; after filtering {}".format(
                1. * _ct_src[1] / sum(_ct_src) if sum(_ct_src) > 0 else 0.,
                1. * _ct_tgt[1] / sum(_ct_tgt) if sum(_ct_tgt) > 0 else 0.
            ))

class BaseProcessor(object):
    # dict_e = dict((k, spacy_tokenize(v)) for k, v in load_json("data/kb/items_wikidata_n.json").items())
    dict_e = load_json("data/kb/items_wikidata_n_tokenized.json")
    dict_p = dict((k, spacy_tokenize(v)) for k, v in load_json("data/kb/filtered_property_wikidata4.json").items())
    dict_t2e = load_json("data/kb/par_child_dict.json")
    dict_e2t = dict((v, k) for k, vs in dict_t2e.items() for v in vs)

    def __init__(self):
        self._labels_dict = None  # label_name: {"type":str, "labels":list}
        self.primary_metric = None

    @staticmethod
    def post_process_dialog_turn(dialog_turn, *args, **kwargs):
        return dialog_turn

    def get_labels_dict(self):
        assert self._labels_dict is not None, "using labels before generation"
        return self._labels_dict


class E2eProcessor(BaseProcessor):
    label_map = {  # from item_type to the name in labels_dict
        "entities": "entities",
        "predicates": "predicates",
        "types": "types",
        "EOs": "EOs",
        "entity_types": "types",
        "sketch": "sketch",
    }

    def __init__(self):
        super(E2eProcessor, self).__init__()
        self.dict_qt2sketches = None

    def transform_example_to_feature(self, example, labels_dict, tokenizer):
        # for utterance
        example["tokenized_utterances_ids"] = {}
        example["tokenized_utterances_pos_ids"] = {}
        for u_type, tokenized_utterance in example["tokenized_utterances"].items():
            example["tokenized_utterances_ids"][u_type] = []
            example["tokenized_utterances_pos_ids"][u_type] = []
            for pos_id, token_text in enumerate(tokenized_utterance.split()):
                ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token_text))
                example["tokenized_utterances_ids"][u_type].extend(ids)
                example["tokenized_utterances_pos_ids"][u_type].extend([pos_id] * len(ids))

        # for labels
        for item_type in ["predicates", "types", "EOs", "entity_types"]:  # "entities",
            label_name = self.label_map[item_type]
            label_list = labels_dict[label_name]["labels"]
            item_id_type = item_type + "_ids"
            example[item_id_type] = {}
            for u_type, vals in example[item_type].items():
                example[item_id_type][u_type] = [label_list.index(val) if val in label_list else 0 for val in vals]

        # for digitized sketch
        if "lf" in example: # target: ["lf"]["gold_sketch"] ["gold_leaves"]
            example["lf"]["gold_sketch_ids"] = []
            example["lf"]["gold_leaves_ids"] = []
            if example["lf"]["gold_sketch"] is not None:
                example["lf"]["gold_sketch_ids"] = [
                    self._labels_dict["sketch"]["labels"].index(_token) for _token in example["lf"]["gold_sketch"]]
                assert len(example["lf"]["gold_sketch"]) == len(example["lf"]["gold_leaves"])
                for _skt_token, _leaf in zip(example["lf"]["gold_sketch"], example["lf"]["gold_leaves"]):
                    if _leaf is None or _skt_token in ["e", "num_utterence"]:
                        example["lf"]["gold_leaves_ids"].append(_leaf)
                    elif _skt_token == "r":
                        example["lf"]["gold_leaves_ids"].append(self._labels_dict["predicates"]["labels"].index(_leaf))
                    elif _skt_token == "Type":
                        example["lf"]["gold_leaves_ids"].append(self._labels_dict["types"]["labels"].index(_leaf))
                    else:
                        assert AttributeError, "{}, {}".format(
                            example["lf"]["gold_sketch"], example["lf"]["gold_leaves"])
        return example

    def get_updated_example_list(self, example_list_sets, update_labels=False, **kwargs):
        # the 1st set in example_list_sets is train
        logging.info("\tget example list in the processor")

        if self._labels_dict is None:
            assert update_labels
        logging.info("\t\tpost processing the example list")
        new_example_sets = []
        for idx_s, example_list in enumerate(example_list_sets):
            new_example_list = []
            for example in example_list:
                new_example_list.append(self.post_process_dialog_turn(example))
            new_example_sets.append(new_example_list)
        example_list_sets = new_example_sets

        if update_labels:
            logging.info("\t\tupdating all the labels")
            special_tokens = [PAD_TOKEN, EMPTY_TOKEN, UNK_TOKEN]
            self._labels_dict = {
                "entities": {
                    "labels": special_tokens + list(BaseProcessor.dict_e.keys()),
                    "type": "",
                },
                "predicates": {
                    "labels": special_tokens + list(BaseProcessor.dict_p.keys()),
                    "type": "",
                },
                "types": {
                    "labels": special_tokens + list(BaseProcessor.dict_t2e.keys()),
                    "type": "",
                },
                "EOs": {
                    "labels": [PAD_TOKEN, "O", "S", "B", "E", "M"],
                    "type": "",
                },
                "sketch": {
                    "labels": [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + [
                        "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14",
                        "A15", "A22", "A23", "A24", "A25", "A26", "A27", "A28", ] + ['e', 'r', "Type", "num_utterence"],
                    "type": "",
                },
            }
            # update valid sketch list wrt question type
            train_example_list = example_list_sets[0]
            self.dict_qt2sketches = self._get_valid_sketches_wrt_question_type(train_example_list)

        logging.info("\t\tprocessing the examples after updating labels")
        new_example_sets = []
        for idx_s, example_list in enumerate(example_list_sets):
            ul = idx_s == 0 and update_labels  # the flag to update labels from
            new_example_list = []
            for example in example_list:
                new_example_list.append(self.process_dialog_turn(example))
            new_example_sets.append(new_example_list)
        return new_example_sets

    @staticmethod
    def post_process_dialog_turn(dialog_turn, *args, **kwargs):
        if "lf" in dialog_turn:  # for training and dev
            # 1. filtering by the occurrence of entity and predicates
            true_lfs = dialog_turn["lf"]["true_lf"]
            # 1.1 gold ent
            gold_entities = set(dialog_turn["entities"]["cur_q"])
            # 1.2 gold predicates
            gold_predicates = set(dialog_turn["predicates"]["cur_q"])
            # 1.3 gold num
            num2idxs = index_num_in_tokenized_utterance(
                dialog_turn["tokenized_utterances"]["cur_q"],
                [eo_label != "O" for eo_label in dialog_turn["EOs"]["cur_q"]])
            gold_nums = set(num2idxs.keys())
            # if len(gold_nums) > 0:
            #     print("QT: {}; len: {}, utterance: {}; nums: {}".format(
            #         dialog_turn["question_type"],
            #         len(gold_nums), dialog_turn["utterances"]["cur_q"], gold_nums))
            valid_indices = []
            for idx_lf, true_lf in enumerate(true_lfs):
                entities_in_lf = set()
                predicates_in_lf = set()
                num_in_lf = set()
                for _item_type, _item_code in true_lf[1]:
                    if _item_type == "e":
                        entities_in_lf.add(_item_code)
                    elif _item_type == "r":
                        predicates_in_lf.add(_item_code)
                    elif _item_type == "num_utterence":
                        num_in_lf.add(int(_item_code))
                # if gold_entities.issubset(entities_in_lf) and gold_predicates.issubset(predicates_in_lf):
                #     valid_indices.append(idx_lf)
                if gold_entities == entities_in_lf:
                    if gold_predicates.issubset(predicates_in_lf):
                        if gold_nums == num_in_lf:
                            valid_indices.append(idx_lf)
                            cond = 0
                        else:
                            cond = 1
                    else:
                        cond = 2
                else:
                    cond = 3
                # if cond != 0:
                #     print("\t\tcond", cond)
            # print("corrct {} / {}: {}".format(len(valid_indices), len(true_lfs), len(valid_indices)>0))
            dialog_turn["lf"]["valid_indices"] = valid_indices
        return dialog_turn

    def process_dialog_turn(self, dialog_turn):
        if "lf" in dialog_turn:  # for training and dev
            # 1. assign sketches to example
            true_lfs = dialog_turn["lf"]["true_lf"]
            true_lfs = [true_lfs[_idx] for _idx in dialog_turn["lf"]["valid_indices"]]
            lf_sketches = E2eProcessor._get_lf_sketches(true_lfs)
            lf_sketches_set = set(lf_sketches)
            qt_valid_sketches = self.dict_qt2sketches[dialog_turn["question_type"]]
            resulting_sketches = []
            for _qt_v_sketch in qt_valid_sketches:
                if _qt_v_sketch in lf_sketches_set:
                    resulting_sketches.append(_qt_v_sketch)
            dialog_turn["lf"]["sketches"] = resulting_sketches

            # 2.1 choose the lf for training
            # 2.2 add e2e labels: Entity (e), Predicate (r), Type (t) and Num in utterance (num_in_utterence)
            # 2.3 in case of that the cannot find the gold label in the first sketch, traverse it!
            # 2.4 target: ["lf"]["gold_sketch"],["lf"]["gold_lf"],["lf"]["gold_leaves"]
            dialog_turn["lf"]["gold_sketch"], dialog_turn["lf"]["gold_lf"], dialog_turn["lf"]["gold_leaves"] = \
                None, None, None
            if len(dialog_turn["lf"]["sketches"]) > 0:
                num2idxs = index_num_in_tokenized_utterance(
                    dialog_turn["tokenized_utterances"]["cur_q"],
                    [eo_label != "O" for eo_label in dialog_turn["EOs"]["cur_q"]])
                for _candi_skt in dialog_turn["lf"]["sketches"]:  # traverse the ordered sketch candidate
                    for _lf_skt, _lf in zip(lf_sketches, true_lfs):  # traverse all true LFs
                        if _candi_skt != _lf_skt:
                            continue
                        gold_leaves = []
                        flag4error = False
                        for _skt_tk, (_skt_tk2, _lf_tk) in zip(_candi_skt, _lf[1]):
                            if _skt_tk == "e":
                                ent_code = _lf_tk
                                ent_idx = None
                                ent_appeared = None
                                for _ut in ["cur_q", "prev_a", "prev_q"]:
                                    _dict_ent2idxss = dialog_turn["ent2idxss"][_ut]
                                    if ent_code in _dict_ent2idxss:  # list of index list
                                        ent_appeared = _ut
                                        if len(_dict_ent2idxss[ent_code]) > 0:
                                            ent_idx = _dict_ent2idxss[ent_code][-1][0]
                                            break
                                if ent_idx is None:
                                    flag4error = True
                                    break
                                gold_leaves.append((ent_idx, ent_appeared))
                            elif _skt_tk == "r":
                                gold_leaves.append(_lf_tk)
                            elif _skt_tk == "Type":
                                gold_leaves.append(_lf_tk)
                            elif _skt_tk == "num_utterence":
                                num_int = _lf_tk
                                if num_int in num2idxs:
                                    num_idx = num2idxs[num_int][-1]
                                    gold_leaves.append(num_idx)
                                else:
                                    flag4error = True
                                    break
                            else:
                                gold_leaves.append(None)

                        if flag4error:
                            continue
                        else:
                            dialog_turn["lf"]["gold_sketch"] = _candi_skt
                            dialog_turn["lf"]["gold_lf"] = _lf
                            dialog_turn["lf"]["gold_leaves"] = gold_leaves
                            break
                    if dialog_turn["lf"]["gold_sketch"] is not None:
                        break
                # end of for iter
        return dialog_turn

    @staticmethod
    def _get_valid_sketches_wrt_question_type(example_list, threshold=0.05):
        def _bubble_sort_for_sketch_freq_list(_sketch_freq_list):
            _num_skt = len(_sketch_freq_list)
            for _idx_out in range(1, _num_skt):
                for _idx_in in range(_idx_out, 0, -1):
                    if len(_sketch_freq_list[_idx_in][0]) < len(_sketch_freq_list[_idx_in - 1][0]):
                        _sketch_freq_list[_idx_in], _sketch_freq_list[_idx_in - 1] = \
                            _sketch_freq_list[_idx_in - 1], _sketch_freq_list[_idx_in]
                    else:
                        break
            return _sketch_freq_list


        dict_id2sketches = {}
        dict_sketch2ids = {}
        dict_qt2ids = {}
        dict_id2qt = {}
        for _id_ex, _example in enumerate(example_list):
            question_type = _example["question_type"]
            dict_id2qt[_id_ex] = question_type
            if question_type not in dict_qt2ids:
                dict_qt2ids[question_type] = []
            dict_qt2ids[question_type].append(_id_ex)

            true_lfs = _example["lf"]["true_lf"]
            valid_indices = _example["lf"]["valid_indices"]
            true_lfs = [true_lfs[_idx] for _idx in valid_indices]
            # get lf sketches
            lf_sketches = E2eProcessor._get_lf_sketches(true_lfs)

            # use dict
            dict_id2sketches[_id_ex] = lf_sketches
            lf_sketches = list(set(lf_sketches))  # remove the duplication
            for _sketch in lf_sketches:
                if _sketch not in dict_sketch2ids:
                    dict_sketch2ids[_sketch] = []
                dict_sketch2ids[_sketch].append(_id_ex)

        dict_sketch2ids_wrt_qt = {}
        logging.info("split sketch2ids to specific question type")
        for _qt_str, _qt_ids in dict_qt2ids.items():
            _qt_ids = set(_qt_ids)
            dict_sketch2ids_wrt_qt[_qt_str] = {}
            for _sketch, _ids in dict_sketch2ids.items():
                dict_sketch2ids_wrt_qt[_qt_str][_sketch] = list(_qt_ids.intersection(_ids))

        # remove zero sketch and sort to list
        sorted_sketch2ids_wrt_qt = {}
        for _qt_str, _sketch2ids in dict_sketch2ids_wrt_qt.items():
            for _sketch in list(_sketch2ids.keys()):  # remove the sketch which contain zero id
                if len(_sketch2ids[_sketch]) == 0:
                    _sketch2ids.pop(_sketch)
            _sketch_freq_list = list(sorted(_sketch2ids.items(), key=lambda d: len(d[1]), reverse=True))
            sorted_sketch2ids_wrt_qt[_qt_str] = _sketch_freq_list

        sketch2ids_wrt_qt = sorted_sketch2ids_wrt_qt

        # further sort
        threshold_num = 4
        verbose = True
        new_sketch2ids_wrt_qt = {}
        for _qt_str, _sketch_freq_list in sketch2ids_wrt_qt.items():
            if verbose:
                _cover_num_all = len(E2eProcessor._calc_cover_elems(_sketch_freq_list))
                logging.info("============ {} =========== {}".format(_qt_str, _cover_num_all))
            new_sketch2ids_wrt_qt[_qt_str] = []
            if len(_sketch_freq_list) == 0:
                continue
            if "Simple" in _qt_str:
                # bubble sort
                _sketch_freq_list = _bubble_sort_for_sketch_freq_list(_sketch_freq_list)
                _sketch_freq_list = [(_sketch, _id_list) for _sketch, _id_list in _sketch_freq_list if
                                     len(_id_list) > threshold_num]
                new_sketch2ids_wrt_qt[_qt_str] = _sketch_freq_list
                if verbose:
                    for _sketch, _q_ids in _sketch_freq_list:
                        logging.info("  All Cover Number: {}; {}".format(len(_q_ids), _sketch))
            else:
                _cover_ids = E2eProcessor._calc_cover_elems(_sketch_freq_list)
                _cover_num_all = len(_cover_ids)

                _all_covered_ids = set()
                _num_candidate_sketches = len(_sketch_freq_list)
                _used_indices = []
                _useless_indices = []
                while len(_used_indices) < _num_candidate_sketches:
                    _max_index = -1
                    _max_new_cover_num = -1
                    _max_new_set = None
                    for _idx_c in range(_num_candidate_sketches):
                        if _idx_c in _used_indices:
                            continue
                        _skt_tuple, _c_list = _sketch_freq_list[_idx_c]
                        _c_set = set(_c_list) - _all_covered_ids
                        _c_new_num = len(_c_set)
                        if _c_new_num > _max_new_cover_num:
                            _max_index = _idx_c
                            _max_new_cover_num = _c_new_num
                            _max_new_set = _c_set
                    assert _max_index > -1
                    _used_indices.append(_max_index)
                    if _max_new_cover_num > threshold_num:
                        _all_covered_ids.update(_max_new_set)
                        if verbose:
                            logging.info("  New Cover Number: {}; {}".format(_max_new_cover_num,
                                                                             _sketch_freq_list[_max_index][0]))
                    else:
                        _useless_indices.append(_max_index)

                new_sketch2ids_wrt_qt[_qt_str] = [_sketch_freq_list[_idx]
                                                  for _idx in _used_indices if _idx not in _useless_indices]
                new_sketch2ids_wrt_qt[_qt_str] = _bubble_sort_for_sketch_freq_list(new_sketch2ids_wrt_qt[_qt_str])

        # remove the ids
        sketch2ids_wrt_qt["Clarification"] = []
        filtered_qt2sketch = {}
        for _qt_str, _sketch_freq_list in sketch2ids_wrt_qt.items():
            filtered_qt2sketch[_qt_str] = [_sketch for _sketch, _ in _sketch_freq_list]

        logging.info("The stats for the filtered sketches...")
        all_lens = []
        for _qt_str, _sketch_list in filtered_qt2sketch.items():
            cur_lens = [len(_skt) for _skt in _sketch_list]
            all_lens += cur_lens
            num_skt = len(cur_lens)
            cur_lens = cur_lens if num_skt > 0 else [0]
            logging.info("\tfor {}, min len is {}, max len is {}, avg is {}, std is {}".format(
                _qt_str, min(cur_lens), max(cur_lens), np.mean(cur_lens), np.std(cur_lens)
            ))
        logging.info("\tfor ALL, min len is {}, max len is {}, avg is {}, std is {}".format(
            min(all_lens), max(all_lens), np.mean(all_lens), np.std(all_lens)
        ))

        return filtered_qt2sketch

    @staticmethod
    def _get_lf_sketches(true_lfs):
        lf_sketches = []
        for true_lf in true_lfs:
            tmp_sketch = []
            for _item_type, _item_code in true_lf[1]:
                if _item_type == "Action":
                    tmp_sketch.append(_item_code)
                elif _item_type in ["e", "r", "Type", "num_utterence"]:
                    tmp_sketch.append(_item_type)
                else:
                    raise AttributeError("_item_type is {}".format(_item_type))
            lf_sketches.append(tuple(tmp_sketch))
        return lf_sketches

    @staticmethod
    def _calc_cover_elems(_freq_list):
        _cover_set = set()
        for _, _cov in _freq_list:
            _cover_set.update(_cov)
        return list(_cover_set)


