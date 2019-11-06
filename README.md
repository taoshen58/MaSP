# Multi-Task Learning for Conversational Question Answering over a Large-Scale Knowledge Base


**Release Note:**
* This is an initial releasing version
* Thanks @guoday for providing the codes for grammars and logical form search method  
* Some programs are not tested after moving to this repo and keeping anonymous, so if any bug occurred during running, no hesitate to contact me via tao.shen@student.uts.edu.au

## Requirement

This code is based on Tensorflow (1.11 or 1.10) and implemented via Python 3.6. Please install following python package:
```
timeout_decorator, fuzzywuzzy, tqdm, flask, nltk, ftfy
spacy
numpy
tensorflow-gpu==1.11.0

```


## Download dataset and Bert Model (for its vocab)

To reimplement our experiment, you need to download dataset from [website](https://amritasaha1812.github.io/CSQA/download/).

1. Download and unzip [dialog files](https://drive.google.com/file/d/1dgf-Qjvhfv-_EWoDjrTCAY5CwYCw-djt/view) (**CSQA_v9.zip** ) to the data folder

   ``` 
   shell
   unzip data/CSQA_v9.zip -d data/
   mv data/CSQA_v9 data/CSQA
   ```

2. Download [wikidata](https://drive.google.com/drive/folders/1ITcgvp4vZo1Wlb66d_SnHvVmLKIqqYbR) and  move all wikidata jsons to "data/kb"

   ``` 
   shell
   mkdir data/kb
   ```
 
1. Down load bert base model from the official [github repo](https://github.com/google-research/bert) to any where

## Preprocess and Build KB

Simply run the following code for preprocess and build kb
```
python main_bfs_preprocess.py
```

## Search GOLD Logical form for Training and Dev
Some key parameter are:
```
num_parallel: The number of processes to perform the BFS, each consume about 55G ram
max_train: the number of dialog to search
```

### For the training data
90000 dialogs will be searched
```
python main_bfs_run.py -mode offline -num_parallel 10 -beam_size 1000 -start_index 0 -max_train 90000 -data_mode dir -data_path data/BFS/train -shuffle 1 -out_dir_suffix wo_con -mask_mode direct -all_lf 0
```
### For the dev data 
6000 dialogs will be searched
```
python main_bfs_run.py -mode offline -num_parallel 10 -beam_size 1000 -start_index 0 -max_train 6000 -data_mode dir -data_path data/BFS/dev -shuffle 1 -out_dir_suffix subset_wo_con -mask_mode direct -all_lf 0
```

The resulting bfs results are `data/BFS/train_proc_direct_1000_wo_con` and `data/BFS/dev_proc_direct_1000_subset_wo_con`. 

### Training the multi-task model
key argments:
* pretrained_num_layers: -2 for do not loading any pretrianed and -1 for loading the word emb.
* num_parallels: the number of process to parallally decoding, each consumes 70-80G ram.

```
python3 main_e2e.py --network_class bert --network_type bert_template --dataset e2e_wo_con \
--preprocessing_hparams \
bert_pretrained_dir=path_to_bert_base,max_sequence_len=72 --training_hparams \
train_batch_size=128,test_batch_size=128,num_epochs=8,eval_period=2000,save_model=True \
--model_hparams \
pos_gain=10.,use_qt_loss_gain=True,seq_label_loss_weight=1.,seq2seq_loss_weight=1.5,pretrained_num_layers=-2,level_for_ner=1,level_for_predicate=1,level_for_type=1,level_for_dec=1,decoder_layer=2,warmup_proportion=0.01,learning_rate=1e-4,hidden_size_input=300,num_attention_heads_input=6,intermediate_size_input=1200,hn=300,clf_head_num=6, \
--model_dir_prefix exp \
--gpu 0;
```

This will result in a dir located in `runtime/run_model/xxx` where `xxx` is the model id.


### decoding using trained model
key argments:
* timeout: timeout second for each example
* num_parallels: the number of process to parallally decoding, each consumes 70-80G ram.
* gpu: each gpu with 16G memory can bear maximum three parallels

```
python3 main_e2e.py --mode parallel_test --network_class bert --network_type bert_template \
--dataset e2e_wo_con \
--preprocessing_hparams bert_pretrained_dir=path_to_bert_base,timeout=5.,num_parallels=7,dump_dir=multi_sp,kb_mode=offline,use_filtered_ent=True \
--training_hparams \
load_model=True,load_path=/path_to_the_dir_gen_previously/ckpt \
--model_dir_prefix parallel_decoding \
--gpu 0,1,2
```

## Paper

[Paper download](https://arxiv.org/pdf/1910.05069.pdf)

[Paper Introduction Video](https://drive.google.com/file/d/1Y9XrTdEaOFaFjUxEA4THWL0VhT4sBXXN/view?usp=sharing)

BibTex:

```
@article{shen2019masp,
  author    = {Tao Shen and
               Xiubo Geng and
               Tao Qin and
               Daya Guo and
               Duyu Tang and
               Nan Duan and
               Guodong Long and
               Daxin Jiang},
  title     = {Multi-Task Learning for Conversational Question Answering over a Large-Scale
               Knowledge Base},
  journal   = {CoRR},
  volume    = {abs/1910.05069},
  year      = {2019},
  url       = {http://arxiv.org/abs/1910.05069}
}
```
