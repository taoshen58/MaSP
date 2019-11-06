#!/usr/bin/env python3
#-*-coding=utf-8-*-  
import sys
import os
import json
from nltk.tokenize import WordPunctTokenizer
import numpy as np
import pickle
import numpy
import random
from tqdm import tqdm
import BFS.agent as agent
import random
import BFS.parser as Parser
import threading
import multiprocessing
import time
import timeout_decorator
import argparse
from utils.csqa import load_list_from_file
from utils.csqa import get_data_path_list, get_entities, get_predicates, load_json
from utils.csqa import generate_EO_with_etype, index_num_in_tokenized_utterance
from glob import glob
from utils.spacy_tk import spacy_tokenize


class Memory(object):
    """ Dialog Memory
    keep and update entities and relation (in our code, we also use predicate(pre) to represent relation)
    """

    def __init__(self):
        self.entities = []
        self.pre = []

    def current_state(self, entities, pre):  # read: all arguments are lis
        current_entities = entities + self.entities
        if pre:
            current_pre = pre
        else:
            current_pre = self.pre
        return current_entities, current_pre

    def update(self, entities, pre):
        self.entities = entities
        self.pre = pre

    def clear(self):
        self.entities = []
        self.pre = []


class ActorParsing(multiprocessing.Process):
    def __init__(self,
                 database, files,
                 all_lf,
                 cover_num_True, cover_num_False, verb, beam_size, out_dir_suffix):
        super(ActorParsing, self).__init__()
        self._database = database
        self._files = files
        self._all_lf = all_lf
        self._cover_num_True = cover_num_True
        self._cover_num_False = cover_num_False
        self._verb = verb
        self._beam_size = beam_size
        self._out_dir_suffix = out_dir_suffix

        self._dict_ent2text = None

        self.daemon = True

    def run(self):
        database = self._database
        files = self._files
        cover_num_True = self._cover_num_True
        cover_num_False = self._cover_num_False
        verb = self._verb
        beam_size = self._beam_size

        parser = Parser.Parser(database)
        parser.load_child2parent()
        memory = Memory()
        for f in tqdm(files, total=len(files)):
            # xxx added: output to another dir
            f_dir = os.path.dirname(f)
            new_f_dir = f_dir + "_proc_{}_{}_".format("direct", beam_size) + self._out_dir_suffix
            if not os.path.isdir(new_f_dir):
                os.mkdir(new_f_dir)
            new_f = os.path.join(new_f_dir, os.path.basename(f))
            if os.path.isfile(new_f):
                try:
                    with open(new_f, 'r') as fp:
                        tmp_dicts = json.load(fp)
                        for i in range(0, len(tmp_dicts), 2):
                            # check whether found the correct answer
                            if tmp_dicts[i]["question-type"] not in cover_num_True:
                                cover_num_True[tmp_dicts[i]["question-type"]] = 0.0
                                cover_num_False[tmp_dicts[i]["question-type"]] = 0.0
                            True_lf_action = tmp_dicts[i + 1]["true_lf"]
                            if len(True_lf_action) != 0:
                                cover_num_True[tmp_dicts[i]["question-type"]] += 1
                            else:
                                cover_num_False[tmp_dicts[i]["question-type"]] += 1
                    continue
                except:
                    pass

            # load dataset
            dicts = json.load(open(f, 'r'))
            # reset memory
            memory.clear()
            # print("+++++++++++++++++{}++++++++++++++++++++".format(os.path.basename(f)))
            prev_predicates = []
            for i in range(0, len(dicts), 2):
                turn_start_time = time.time()
                # Extract entity and relation, in BFS, we use entities and relations offered by training dataset
                # In D2A, we only use entities by entity linking and relations by relation classfier
                # In our setting, we assume that entities and relations are unseen in test dataset
                if 'entities_in_utterance' in dicts[i]:
                    user_entities = dicts[i]['entities_in_utterance']
                else:
                    user_entities = []
                if 'entities_in_utterance' in dicts[i + 1]:
                    system_entities = dicts[i + 1]['entities_in_utterance']
                # elif 'entities' in dicts[i + 1]:
                #     system_entities = dicts[i + 1]['entities']
                else:
                    system_entities = []
                if 'relations' in dicts[i]:  # gold relations are used
                    pres = dicts[i]['relations']
                else:
                    pres = []
                if 'type_list' in dicts[i]:
                    types = dicts[i]['type_list']  # gold types are used
                else:
                    types = []
                numbers = []
                for x in dicts[i]['utterance'].split():
                    try:
                        numbers.append(int(x))
                    except:
                        continue
                numbers = list(set(numbers))
                entities, pres = memory.current_state(user_entities, pres)

                # # our method !!!!!!!!!!!!!!!!!!!!!!!!!
                # # 1. for the entity
                entities = get_entities(dicts[i])
                # # 2. for the number: i.e., remove the number in the entity
                if self._dict_ent2text is None:
                    self._dict_ent2text = load_json("data/kb/items_wikidata_n_tokenized.json")
                cur_q_utterance = dicts[i]["utterance"]
                tokenized_utterance = spacy_tokenize(cur_q_utterance)
                ent_codes = entities.copy()
                ent_strs = [self._dict_ent2text[_code] for _code in ent_codes]
                if len(ent_codes) > 0:
                    ent_codes, ent_str = zip(
                        *list(
                            sorted(zip(ent_codes, ent_strs),
                                   key=lambda elem: len(elem[1].split()), reverse=True)
                        ))
                EO, _, _ = generate_EO_with_etype(
                    tokenized_utterance, ent_codes, ent_strs, ["UNK"]*len(ent_codes), "EMPTY")
                num2idxs = index_num_in_tokenized_utterance(tokenized_utterance, [eo_label != "O" for eo_label in EO])
                numbers = list(num2idxs.keys())
                # 3. predicates
                cur_predicates = get_predicates(dicts[i])
                if len(cur_predicates) == 0:
                    pres = prev_predicates
                else:
                    pres = cur_predicates
                prev_predicates = cur_predicates

                # Extract answer
                answer = parser.parsing_answer(dicts[i + 1]['all_entities'], dicts[i + 1]['utterance'],
                                               dicts[i]['question-type'])
                try:
                    logical_forms, candidate_answers, logical_action, _ = parser.BFS(
                        entities, pres, types, numbers, beam_size)  # add set
                except timeout_decorator.TimeoutError:
                    logical_forms = []
                    candidate_answers = []
                    logical_action = []
                    # lf_entity_record = []
                # update memory and keep right logical forms and action sequences
                memory.update(user_entities + system_entities, pres)
                True_lf = []
                True_lf_action = []
                # True_lf_entity_record = []
                All_lf = []
                for item in zip(logical_forms, candidate_answers, logical_action):
                    pred = item[1]
                    All_lf.append(item[0])
                    All_lf.append((item[0], item[2]))
                    if type(pred) == int:
                        pred = [pred]
                    if answer == pred:
                        True_lf.append(item[0])
                        True_lf_action.append((item[0], item[2]))
                        # True_lf_entity_record.append(item[3])

                # eval oracle
                if dicts[i]["question-type"] not in cover_num_True:
                    cover_num_True[dicts[i]["question-type"]] = 0.0
                    cover_num_False[dicts[i]["question-type"]] = 0.0
                if len(True_lf_action) != 0:
                    cover_num_True[dicts[i]["question-type"]] += 1
                else:
                    cover_num_False[dicts[i]["question-type"]] += 1
                dicts[i + 1]["true_lf"] = True_lf_action
                if self._all_lf:
                    dicts[i + 1]['all_lf'] = All_lf
                dicts[i + 1]['num_all_lf'] = len(All_lf)
                dicts[i + 1]['time'] = time.time() - turn_start_time
            json.dump(dicts, open(new_f, 'w'))


def main():
    parser = argparse.ArgumentParser(description='BFS/run.py')
    parser.add_argument('-mode', required=True, help="offline or online")
    parser.add_argument('-num_parallel', type=int, default=1, help="degree of parallelly")
    parser.add_argument('-beam_size', type=int, default=1000, help="beam size of BFS")
    parser.add_argument('-start_index', type=int, default=0, help="start_index for train")
    parser.add_argument('-max_train', type=int, default=60000, help="number of dialogs to search")
    parser.add_argument('-sort', type=int, default=0, help="wheather to shuffle the train")
    parser.add_argument('-shuffle', type=int, default=0, help="wheather to shuffle the train")
    parser.add_argument('-file_path_save', type=str, default=None, help="file_path str save to this file")

    parser.add_argument('-data_mode', type=str, default="none", help="[dir|list|base_list|pattern]data mode")
    parser.add_argument('-data_path', type=str, default="none", help="main path")
    parser.add_argument('-extra_path', type=str, default="none", help="aux path used for base_list mode")

    parser.add_argument('-out_dir_suffix', type=str, default="", help="")

    parser.add_argument('-all_lf', type=int, default=1, help="")

    opt = parser.parse_args()

    # load dataset
    thread_num = opt.num_parallel

    train = []
    if opt.data_mode == "dir":
        train = get_data_path_list("all", opt.data_path)
    elif opt.data_mode == "list":
        train =load_list_from_file(opt.data_path)
    elif opt.data_mode == "base_list":
        train = [os.path.join(opt.extra_path, basename) for basename in load_list_from_file(opt.data_path)]
    elif opt.data_mode == "pattern":
        train = glob(opt.data_path)
    else:
        raise NotImplementedError

    assert len(train) > 0

    # sort
    if opt.sort:
        train = list(sorted(train))

    if opt.shuffle > 0:
        random.seed(opt.shuffle)
        random.shuffle(train)
    print("num train is {}".format(len(train)))
    files = train[opt.start_index:(opt.start_index + opt.max_train)]
    print("First 10 file name:")
    for i in range(5):
        print("    ", files[i])
    if opt.file_path_save is not None:
        with open(opt.file_path_save, "w", encoding="utf-8") as fp:
            for file in files:
                fp.write(os.path.basename(file))
                fp.write(os.linesep)

    manager = multiprocessing.Manager()

    # load knowledge base
    database = agent.KB(mode=opt.mode)

    if opt.max_train != 0:
        # allocating task to different thread
        thread_files = [[] for i in range(thread_num)]
        threads = []
        for idx, f in enumerate(files):
            thread_files[idx % thread_num].append(f)

        # to eval oracle of BFS
        cover_num_True = [manager.dict() for i in range(thread_num)]
        cover_num_False = [manager.dict() for i in range(thread_num)]

        for i in range(thread_num):
            thread = ActorParsing(
                database, thread_files[i],
                opt.all_lf,
                cover_num_True[i], cover_num_False[i], i == 0, opt.beam_size,
                opt.out_dir_suffix,
            )
            thread.start()
            threads.append(thread)
        for t in threads:
            t.join()

        # print result
        cover_num_True = [dict(cover_num_True[i]) for i in range(thread_num)]
        cover_num_False = [dict(cover_num_False[i]) for i in range(thread_num)]
        cover = {}
        for it in cover_num_True[0]:
            cover[it] = [0.0, 0.0]
        for i in range(thread_num):
            for it in cover_num_True[i]:
                try:
                    cover[it][0] += cover_num_False[i][it]
                    cover[it][1] += cover_num_True[i][it]
                except:
                    pass

        # for the sample counter
        counter = {}
        for it in cover_num_True[0]:
            counter[it] = 0
        for i in range(thread_num):
            for it in cover_num_True[i]:
                counter[it] += (cover_num_False[i][it] + cover_num_True[i][it])

        for it in cover:
            print(it, round(cover[it][1] * 100.0 / sum(cover[it]), 2), "({})".format(int(counter[it])))


if __name__ == "__main__":
    main()

