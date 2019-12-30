from utils.csqa import load_json, save_json, get_ent_int_id, save_list_to_file
from fuzzywuzzy import fuzz
import json
from tqdm import tqdm
from utils.spacy_tk import spacy_tokenize


def build_child2parent_list():
    child_par_list = "data/kb/child_par_list.json"
    par_child_dict = load_json("data/kb/par_child_dict.json")
    items_wikidata_n = load_json("data/kb/items_wikidata_n.json")

    max_entity_id = -1
    for ent in items_wikidata_n:
        max_entity_id = max(max_entity_id, get_ent_int_id(ent))
    assert max_entity_id > -1
    # build list
    res_list = [None for _ in range(max_entity_id + 1)]
    for par in par_child_dict:
        for child in par_child_dict[par]:
            res_list[get_ent_int_id(child)] = par
    save_json(res_list, child_par_list)


def tokenize_items_wikidata_n():
    items_wikidata_n = load_json("data/kb/items_wikidata_n.json")
    for key in items_wikidata_n:
        items_wikidata_n[key] = spacy_tokenize(items_wikidata_n[key])
    save_json(items_wikidata_n, "data/kb/items_wikidata_n_tokenized.json")


def build_inverse_index_spacy_token():
    dic=json.load(open('data/kb/items_wikidata_n_tokenized.json','r'))
    #create a dict, given a string as key, return ids of entities with highest score
    inverse_index={}
    count = 0
    for ids in tqdm(dic,total=len(dic)):
        temp=dic[ids].split()
        if count < 100:
            count += 1
            print(dic[ids], temp)
        for i in range(len(temp)):
            for j in range(i+1,len(temp)+1):
                words=' '.join(temp[i:j])
                if j-i+1<=len(temp)-2:
                    continue
                score=fuzz.ratio(words,temp)
                if words not in inverse_index or inverse_index[words]['score']<score:
                    inverse_index[words]={}
                    inverse_index[words]['score']=score
                    inverse_index[words]['idxs']=[ids]
                elif inverse_index[words]['score']==score:
                    inverse_index[words]['idxs'].append(ids)


    json.dump(inverse_index,open('data/EDL/inverse_index_spacy_token.json','w'))


if __name__ == '__main__':
    build_child2parent_list()
