import os
import logging
from collections import Counter


def underline_to_camel(underline_format):
    camel_format = ''
    if isinstance(underline_format, str):
        for _s_ in underline_format.split('_'):
            camel_format += _s_.capitalize()
    return camel_format


def mp_join(*args):
    dir_path = os.path.join(*args)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def get_val_str_from_dict(val_dict):
    # sort
    sorted_list = list(sorted(val_dict.items(), key=lambda item: item[0]))
    str_return = ""
    for key, val in sorted_list:
        if len(str_return) > 0:
            str_return += ", "
        str_return += "%s: %.4f" % (key, val)
    return str_return


def matching_by_voting(src_token_list, tgt_token_list, tgt_attr_list):
    assert len(src_token_list) <= len(tgt_token_list)
    assert len(tgt_token_list) == len(tgt_attr_list)

    src_attr_list = []
    idx_tgt = 0
    for src_token in src_token_list:
        attr_buff = []
        idx_char = 0
        while idx_tgt < len(tgt_token_list):
            idx_char_new = src_token.find(tgt_token_list[idx_tgt], idx_char)
            if idx_char_new < 0:
                logging.warning("For matching_by_voting, src: {}, tgt:{}".format(src_token, str(tgt_token_list)))
                break
            attr_buff.append(tgt_attr_list[idx_tgt])
            idx_char = idx_char_new + len(tgt_token_list[idx_tgt])
            idx_tgt += 1
            if idx_char == len(src_token):
                break
        counter = Counter(attr_buff)
        src_attr_list.append(counter.most_common()[0][0])
    assert len(src_token_list) == len(src_attr_list)
    return src_attr_list


if __name__ == '__main__':
    src_t = ['Rolls-Royce',
             'Motor',
             'Cars',
             'Inc.',
             'said',
             'it',
             'expects',
             'its',
             'U.S.',
             'sales',
             'to',
             'remain',
             'steady',
             'at',
             'about',
             '1,200',
             'cars',
             'in',
             '1990',
             '.']
    tgt_t =['Rolls',
             '-',
             'Royce',
             'Motor',
             'Cars',
             'Inc.',
             'said',
             'it',
             'expects',
             'its',
             'U.S.',
             'sales',
             'to',
             'remain',
             'steady',
             'at',
             'about',
             '1,200',
             'cars',
             'in',
             '1990',
             '.']

    tgt_a = ['PROPN',
             'PUNCT',
             'PROPN',
             'PROPN',
             'PROPN',
             'PROPN',
             'VERB',
             'PRON',
             'VERB',
             'ADJ',
             'PROPN',
             'NOUN',
             'PART',
             'VERB',
             'ADJ',
             'ADP',
             'ADP',
             'NUM',
             'NOUN',
             'ADP',
             'NUM',
             'PUNCT']

    print(matching_by_voting(src_t, tgt_t, tgt_a))























