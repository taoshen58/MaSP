import numpy as np
import os
import json
import pickle
import copy

def get_ent_int_id(idx):
    return int(idx[1:])

def get_data_path_list(data_type="dev", dir=None):
    path_list = []
    for root, dirs, files in os.walk(dir or "data/CSQA"):
        for file in files:
            temp = os.path.join(root, file)
            if '.json' in temp:
                if data_type == "train":
                    if 'train' in temp:
                        path_list.append(temp)
                elif data_type == "dev":
                    if 'valid' in temp or "dev" in temp:
                        path_list.append(temp)
                elif data_type == "test":
                    if 'test' in temp:
                        path_list.append(temp)
                elif data_type == "all":
                    path_list.append(temp)
                else:
                    raise AttributeError
    return path_list


def save_list_to_file(str_list, file_path, use_basename=False):
    with open(file_path, "w", encoding="utf-8") as fp:
        for path_str in str_list:
            fp.write(os.path.basename(path_str) if use_basename else path_str)
            fp.write(os.linesep)


def load_list_from_file(file_path):
    data = []
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as fp:
            for line in fp:
                data.append(line.strip())
    return data


def sort_action_list(action_list):
    return list(sorted(list(action_list), key=lambda elem: int(elem[1:])))


def load_pickle(path):
    with open(path, "rb") as fp:
        data = pickle.load(fp)
        return data


def save_pickle(obj, path):
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def load_json(path):
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
        return data


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp)


def filter_dict(example_dict, threshold):
    to_pop_key_list = []
    for key in example_dict:
        if len(example_dict[key]) < threshold:
            to_pop_key_list.append(key)
    for key in to_pop_key_list:
        example_dict.pop(key)
    return example_dict


def classification_wrt_question_type(turn_list, out_dict=None, threshold=0):
    out_dict = out_dict or {}

    for turn in turn_list:
        assert "question-type" in turn or "question_type" in turn
        qt = turn.get("question-type") or turn.get("question_type")
        assert qt is not None
        if qt not in out_dict:
            out_dict[qt] = []
        out_dict[qt].append(turn)
    return filter_dict(out_dict, threshold)


def transform_turn_list_to_history_format(turn_list):
    example_list = []
    prev_turn = None
    for idx_t in range(0, len(turn_list), 2):
        cur_turn = {
            "user": copy.deepcopy(turn_list[idx_t]),
            "system": copy.deepcopy(turn_list[idx_t + 1]),
        }
        example = {
            "question_type": cur_turn["user"]["question-type"],
            "previous": prev_turn,
            "current": cur_turn,
        }
        example_list.append(example)
        prev_turn = cur_turn
    return example_list


# def data_shred_for_efficiency(data_list, start_time, process_speed, num_parallels):
#     N = len(data_list)  # data number
#     delta_n = int(start_time / process_speed)  # delta n for next shred
#     # calc
#     all_start_time = (0. + (num_parallels-1.))*num_parallels/2. #
#     all_process_time = N * process_speed
#     single_thread_time = (all_start_time + all_process_time) / num_parallels
#     first_shred_len = single_thread_time / process_speed
#
#     # reversed load data
#     data_shred_list = [[] for _ in range(num_parallels)]
#     data_shred_len = [max(first_shred_len - idx*delta_n, 1) for idx in range(num_parallels)]
#     # balance
#
#
# def data_shred(data_list, num_parallels):
#     data_shred_list = [[] for _ in range(num_parallels)]
#     for idx_d, example in enumerate(data_list):
#         data_shred_list[idx_d%num_parallels].append(example)
#     return data_shred_list


def get_utterance(_dialog_elem):
    if _dialog_elem is None:
        return ""
    return _dialog_elem.get("utterance") or ""


def get_tokenized_utterance(_dialog_elem):
    if _dialog_elem is None:
        return ""
    return _dialog_elem.get("tokenized_utterance") or ""


def get_entities(_dialog_elem):
    if _dialog_elem is None:
        return []
    return _dialog_elem.get("entities_in_utterance") or _dialog_elem.get("entities") or []


def get_predicates(_dialog_elem):
    if _dialog_elem is None:
        return []
    return _dialog_elem.get("relations") or []


def get_types(_dialog_elem):
    if _dialog_elem is None:
        return []
    return _dialog_elem.get("type_list") or []


def get_EOs(_dialog_elem):
    if _dialog_elem is None:
        return []
    return _dialog_elem.get("EOs") or []


def get_entity_types(_dialog_elem):
    if _dialog_elem is None:
        return []
    return _dialog_elem.get("entity_types") or []


def get_numbers(_dialog_elem):
    n = []
    for x in _dialog_elem['utterance'].split():
        try:
            n.append(int(x))
        except:
            continue
    n = list(set(n))
    return n


# ============= EO and entity type label
def replace_EO(sentence, EO, ent_type_labels, cur_type, entities):
    s = ' '.join(sentence)
    e = ' '.join(entities)
    s = s.replace(e, ' '.join(['XXX'] * len(entities)))
    s = s.split()
    assert len(s) == len(EO)
    flag = True  # out of an entity
    cont = 0
    indices_list = []
    for i in range(len(s)):
        if s[i] == 'XXX':
            s[i] = 'YYY'
            if flag:
                cont += 1
                flag = False
                if len(entities) == 1:
                    EO[i] = 'S'
                    ent_type_labels[i] = cur_type
                    flag = True
                else:
                    EO[i] = 'B'
                    ent_type_labels[i] = cur_type
                indices_list.append([i])
            else:
                cont += 1
                if cont == len(entities):
                    EO[i] = 'E'
                    ent_type_labels[i] = cur_type
                    flag = True
                else:
                    EO[i] = 'M'
                    ent_type_labels[i] = cur_type
                indices_list[-1].append(i)
    return s, EO, ent_type_labels, indices_list


def generate_EO_with_etype(sentence, entity_codes, entities_in_utterance, entity_types, empty_token):
    # entities_in_utterance is a str list
    s = sentence.split()
    EO = ['O' for _ in s]
    ent_type_labels = [empty_token for _ in s]
    dict_code2indices_list = {}
    for e_code, e, cur_type in zip(entity_codes, entities_in_utterance, entity_types):
        s, EO, ent_type_labels, indices_list = replace_EO(s, EO, ent_type_labels, cur_type, e.split())
        if e_code not in dict_code2indices_list:
            dict_code2indices_list[e_code] = []
        dict_code2indices_list[e_code].extend(indices_list)

    # sort for dict_code2indices_list
    for _code in dict_code2indices_list:
        dict_code2indices_list[_code] = list(
            sorted(dict_code2indices_list[_code], key=lambda elem: elem[0])  # the first index
        )

    return EO, ent_type_labels, dict_code2indices_list


def index_num_in_tokenized_utterance(tokenized_utterance, ent_mask=None):
    tk_list = tokenized_utterance.split()
    if ent_mask is None:
        ent_mask = [False] * len(tk_list)
    assert len(tk_list) == len(ent_mask)

    num2idxs = {}
    for _idx_t, _tk in enumerate(tk_list):
        if ent_mask[_idx_t]:
            continue
        try:
            num = int(_tk)
            if num not in num2idxs:
                num2idxs[num] = []
            num2idxs[num].append(_idx_t)
        except ValueError:
            pass
    return num2idxs









