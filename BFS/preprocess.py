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
from BFS.preproc_fn import build_child2parent_list, tokenize_items_wikidata_n, build_inverse_index_spacy_token


def get_id(idx):
    return int(idx[1:])


def preprocess_data():
    print("Preprocessing data...")
    print("build_child2parent_list")
    build_child2parent_list()
    print("tokenize_items_wikidata_n")
    tokenize_items_wikidata_n()
    print("build_inverse_index_spacy_token")
    build_inverse_index_spacy_token()
    print("index the raw qa files into data/BFS/train, data/BFS/dev and data/BFS/test")
    # load dataset
    train = []
    dev = []
    test = []
    for root, dirs, files in os.walk("data/CSQA"):
        for file in files:
            temp = os.path.join(root, file)
            if '.json' in temp:
                if 'train' in temp:
                    train.append(temp)
                elif 'valid' in temp:
                    dev.append(temp)
                elif 'test' in temp:
                    test.append(temp)

    # entity linking and relation predication
    # train=random.sample(train,min(60000,len(train)))
    for files in [('train', train), ('dev', dev), ('test', test)]:
        print("doing for {}".format(files[0]))
        cont = 0
        for f in tqdm(files[1], total=len(files[1])):
            dicts = json.load(open(f, 'r'))
            sentences = []
            for d in dicts:
                sentences.append(d['utterance'])  # both USER and SYSTEM
            if len(sentences) == 0:
                continue
            if not os.path.exists('data/BFS/' + files[0]):
                os.makedirs('data/BFS/' + files[0])
            json.dump(dicts, open('data/BFS/' + files[0] + '/QA' + str(cont) + '.json', 'w'))
            cont += 1
    print("All done!")


def create_entity_type():
    print("Creating entity_type dictionary...")
    '''
    Build a dictionary
    key: ids of entity
    values: ids of type
    '''

    dic=json.load(open('data/kb/par_child_dict.json'))
    max_id=0
    for d in tqdm(dic,total=len(dic)):
        for idx in dic[d]:
                max_id=max(max_id,get_id(idx))

    type_dict =['' for i in range(max_id+1)]
    for d in dic:
        for idx in dic[d]:
                type_dict[get_id(idx)]=d
    pickle.dump(type_dict,open('data/BFS/type_kb.pkl','wb'))
    
    return type_dict

def create_relation_type(type_dict,path):
    print("Creating relation_type dictionary...")
    '''
    Build a dictionary
    key: ids of relation
    values: set of type that there's a entity belong to this type and link to the relation (key)
    '''
    dic={}  # a buffer, a dict of set: key-pid,value-set
    for f in path:
        dic_kb=json.load(open(f,'r'))
        for idx in tqdm(dic_kb,total=len(dic_kb)):
            try:
                idx_type=type_dict[get_id(idx)]
            except:
                continue
            for p in dic_kb[idx]:  # p is pid, and dic_kb[idx][p] is a list of object
                if p not in dic:
                    dic[p]=set()
                dic[p].add(idx_type)    
                for y in dic_kb[idx][p]: 
                    try:
                        y_type=type_dict[get_id(y)]
                    except:
                        continue  
                    dic[p].add(y_type) 
    pickle.dump(dic,open('data/BFS/type_predicate_link.pkl','wb'))
    
def create_type_relation_type(type_dict,paths):
    print("Creating type_relation_type dictionary...")
    '''
    Build a dictionary
    key: type _x,direction _t, relation _r, type _y
    values: set of entity ids _e
    information abot this: 
    if direction _t="obj" , return all entity ids _e with having the triple (entity _e with _x type, relation(_r), any one of entity with _y type) 
    if direction _t="sub" , return all entity ids _e with having the triple (any one of entity with _y type, relation(_r), entity _e with _x type) 
    '''
    dic={}
    for f in paths:
        dic_kb=json.load(open(f[0],'r'))
        obj=f[1]
        sub=f[2]
        for idx in tqdm(dic_kb,total=len(dic_kb)):
            try:
                idx_type=type_dict[get_id(idx)]
            except:
                continue
            if idx_type=='':
                continue
            if idx_type not in dic:
                dic[idx_type]={}
                dic[idx_type][obj]={}
                dic[idx_type][sub]={}
            for p in dic_kb[idx]:
                if (obj,p,idx_type) not in dic:
                    dic[(obj,p,idx_type)]=set()  # contains the idx belonging to "idx_type"
                dic[(obj,p,idx_type)].add(idx)
                if p not in dic[idx_type][sub]:
                    dic[idx_type][sub][p]={}
                for y in dic_kb[idx][p]: # a list of objects
                    try:
                        y_type=type_dict[get_id(y)]  # objects' entity type
                    except:
                        continue
                    if y_type=="":
                        continue
                    if y_type not in dic[idx_type][sub][p]:
                        dic[idx_type][sub][p][y_type]={}
                    if y not in dic[idx_type][sub][p][y_type]:
                        dic[idx_type][sub][p][y_type][y]=set()
                    if idx!=y:
                        dic[idx_type][sub][p][y_type][y].add(idx)
                    # === for "y_type" as the entry
                    if y_type not in dic:
                        dic[y_type]={}
                        dic[y_type][obj]={}
                        dic[y_type][sub]={}  
                    if p not in dic[y_type][obj]:
                        dic[y_type][obj][p]={}
                    if idx_type not in dic[y_type][obj][p]:
                        dic[y_type][obj][p][idx_type]={}
                    if idx not in dic[y_type][obj][p][idx_type]:
                        dic[y_type][obj][p][idx_type][idx]=set()
                    if idx!=y:
                        dic[y_type][obj][p][idx_type][idx].add(y)  
    for x in dic:
        if type(x)==tuple:
            continue
        for p in dic[x]['sub']:
            for y in dic[x]['sub'][p]:
                temp=[]
                for idx in dic[x]['sub'][p][y]:
                    temp.append((idx,dic[x]['sub'][p][y][idx]))  # (entity y, set(entity of type x))
                dic[x]['sub'][p][y]=temp  # from [type_x][sub][predicate][type_y][entity y ]--[set of type x] =to=>[type_x][sub][predicate][type_y] -- list of (entity y, set(entity x))

    for x in dic:
        if type(x)==tuple:
            continue
        for p in dic[x]['obj']:
            for y in dic[x]['obj'][p]:
                temp=[]
                for idx in dic[x]['obj'][p][y]:
                    temp.append((idx,dic[x]['obj'][p][y][idx]))
                dic[x]['obj'][p][y]=temp

    pickle.dump(dic,open('data/BFS/pre_type.pkl','wb'))
    
def preprocess_kb():

    type_dict=create_entity_type()  # a list: idx is the child id and the value is the parent entity (type)
    create_relation_type(type_dict,["data/kb/wikidata_short_1.json","data/kb/wikidata_short_2.json",
                                    "data/kb/comp_wikidata_rev.json"])
    create_type_relation_type(type_dict,[["data/kb/wikidata_short_1.json","obj","sub"],["data/kb/wikidata_short_2.json","obj","sub"],["data/kb/comp_wikidata_rev.json","sub","obj"]])


def main():
    # #preprocess_data
    preprocess_data()
    # #preprocess data format of knowledge base
    preprocess_kb()
    #
    # create knowledge base
    print("Create knowlege base...")
    agent.create_kb('data/kb')

if __name__ == "__main__":
    # #preprocess_data
    preprocess_data()
    # #preprocess data format of knowledge base
    preprocess_kb()
    #
    #create knowledge base
    print("Create knowlege base...")
    agent.create_kb('data/kb')


