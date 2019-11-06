import tensorflow as tf
import numpy as np
from os.path import join
import os
import logging
from peach.utils.hparams import HParamsCenter, HParams
from peach.utils.file import load_file, save_file
from peach.utils.numeral import MovingAverageDict
from peach.utils.string import get_val_str_from_dict
from peach.tf_nn.utils.graph_handler import GraphHandler, calc_training_step
from peach.tf_nn.utils.performance_recoder import PerformanceRecoder

from utils.csqa import get_data_path_list, load_json, save_json

from e2e.configs import Configs
from e2e.e2e_dataset import E2eDataset, BaseProcessor
from e2e.e2e_evaluator import E2eEvaluator
from e2e.e2e_parallel import parallel_test


def train(cfg):
    loaded_data = load_file(cfg['processed_path'], 'processed_datasets', mode='pickle') if cfg['load_preproc'] else None
    if loaded_data is None or not cfg['load_preproc']:
        dataset_obj = E2eDataset(
            cfg['dataset'], [cfg["train_data_path"], cfg["dev_data_path"]],
            cfg['vocab_file'], cfg['do_lower_case'], labels_dict_for_infer=None,

        )
        dataset_obj.process_training_data()
        dataset_obj.clean_data(del_lfs=True)
        save_file(dataset_obj, cfg['processed_path'])
        # for processed light
        tmp_train_feature_list, tmp_dev_feature_list = dataset_obj._train_feature_list, dataset_obj._dev_feature_list
        dataset_obj._train_feature_list, dataset_obj._dev_feature_list =\
            dataset_obj._train_feature_list[:100], dataset_obj._dev_feature_list[:2000]
        save_file(dataset_obj, cfg['processed_path']+".light")
        dataset_obj._train_feature_list, dataset_obj._dev_feature_list = tmp_train_feature_list, tmp_dev_feature_list
    else:
        dataset_obj = loaded_data
    dataset_obj.log_lf_success_ratio_after_filtering()
    dataset_obj._dev_feature_list = dataset_obj._dev_feature_list[:5000]
    cfg["data_info"] = {
        "labels_dict": dataset_obj.get_labels_dict(),
    }
    cfg.save_cfg_to_file(cfg["cfg_path"])

    # neural model
    num_training_steps = calc_training_step(
        cfg['num_epochs'], cfg['train_batch_size'], dataset_obj.num_train_examples, num_steps=cfg['num_steps'])
    with tf.variable_scope('model') as scope:
        # cfg, vocab, data_type, labels_dict, max_sequence_len, num_training_steps, scope
        model_obj = cfg['model_class'](
            cfg, dataset_obj.tokenizer, cfg['dataset'],
            dataset_obj.get_labels_dict(), cfg["max_sequence_len"], num_training_steps, scope.name
        )
        model_obj.load_pretrained_model_init(num_layers=cfg['pretrained_num_layers'])

    graph_handler = GraphHandler(model_obj, cfg)
    evaluator = E2eEvaluator(model_obj, cfg)
    performance_recorder = PerformanceRecoder(cfg["ckpt_dir"], cfg["save_model"], cfg["save_num"])
    sess = graph_handler.initialize()

    moving_average_dict = MovingAverageDict()
    global_step_val = 0

    for example_batch, batch_num, data_round, idx_b in \
            dataset_obj.generate_batch_iter(cfg['train_batch_size'], 'train', num_training_steps):
        global_step_val += 1
        step_out = model_obj.step(sess, example_batch)
        moving_average_dict(step_out)
        if global_step_val % 100 == 0:
            logging.info('data round: %d: %d/%d, global step:%d -- %s' %
                         (data_round, idx_b, batch_num, global_step_val, moving_average_dict.get_val_str()))

        #  evaluation
        if global_step_val % cfg['eval_period'] == 0 or global_step_val == num_training_steps:
            dev_res = evaluator.get_evaluation(sess, dataset_obj, "dev", global_step_val)
            logging.info('==> for dev, {}'.format(get_val_str_from_dict(dev_res)))
            performance_recorder.update_top_list(global_step_val, dev_res["key_metric"], sess)


def infer(cfg, infer_cfg):
    dataset_obj = load_file(cfg['processed_path'] + ".light", 'processed_datasets', mode='pickle')  #
    assert dataset_obj is not None
    # dataset_obj._dev_feature_list = dataset_obj._dev_feature_list[:2000]

    with tf.variable_scope('model') as scope:
        # cfg, vocab, data_type, labels_dict, max_sequence_len, num_training_steps, scope
        model_obj = cfg['model_class'](
            cfg, dataset_obj.tokenizer, cfg['dataset'],
            dataset_obj.get_labels_dict(), cfg["max_sequence_len"], 1000, scope.name
        )
    graph_handler = GraphHandler(model_obj, infer_cfg)
    evaluator = E2eEvaluator(model_obj, infer_cfg)
    sess = graph_handler.initialize()
    dev_res = evaluator.get_evaluation(sess, dataset_obj, "dev", None)
    logging.info('==> for dev, {}'.format(get_val_str_from_dict(dev_res)))


def decoding(model_cfg, infer_cfg):  # this for dev set
    from e2e.exe import LfExecutor

    dataset_obj = load_file(infer_cfg['processed_path'] + ".light", 'processed_datasets', mode='pickle')  #
    assert dataset_obj is not None
    dataset_obj._dev_feature_list = dataset_obj._dev_feature_list[:4000]

    with tf.variable_scope('model') as scope:
        # cfg, vocab, data_type, labels_dict, max_sequence_len, num_training_steps, scope
        model_obj = model_cfg['model_class'](
            model_cfg, dataset_obj.tokenizer, model_cfg['dataset'],
            dataset_obj.get_labels_dict(), model_cfg["max_sequence_len"], 1000, scope.name
        )
    graph_handler = GraphHandler(model_obj, infer_cfg)
    evaluator = E2eEvaluator(model_obj, infer_cfg)
    sess = graph_handler.initialize()
    # data preparation
    logging.info("loading inverse_index...")
    inverse_index = load_json("data/EDL/inverse_index_spacy_token.json")
    logging.info("building lf executor")
    lf_executor = LfExecutor(kb_mode="offline")
    logging.info("Done")

    evaluator.decoding(
        sess, dataset_obj._dev_feature_list, lf_executor,
        inverse_index, BaseProcessor.dict_e2t,
        dataset_obj.get_labels_dict()["EOs"]["labels"],
        dataset_obj.get_labels_dict()["sketch"]["labels"],
        dataset_obj.get_labels_dict()["predicates"]["labels"],
        dataset_obj.get_labels_dict()["types"]["labels"],
        batch_size=20, max_seq_len=infer_cfg["max_sequence_len"]
    )


def main(_):
    hparams_center = HParamsCenter(
        HParams(  # preprocessing
            load_preproc=True,
            bert_pretrained_dir='none',
            max_sequence_len=64,
            src_infer_dir='none',
            tgt_infer_dir='none',
            # ====== For Decoding ======
            timeout=5.,
            # for knowledge base OP
            use_op_type_constraint=False,
            # for NER
            ner_dump_dir="save_ner_num",
            # for decoding
            debug_dec=0,
            num_parallels=4,
            dump_dir="placeholder",
            clear_dump_dir=False,
            kb_mode="online",
            verbose_test=False,
            use_filtered_ent=True,
            use_dump_ner=False,
        ),
        HParams(  # model
            pretrained_num_layers=-1,
        ),
        HParams(  # training
            num_epochs=3,
            num_steps=-1,
            train_batch_size=32,
            test_batch_size=32,

            load_model=False,
            load_path='none',

            eval_period=500,
            save_model=False,
            save_num=3,
        ), models_dir="e2e.models"
    )
    cfg = Configs(hparams_center, 'multi_task_sp')

    if cfg['mode'] == 'train':
        train(cfg)
    elif cfg['mode'] == 'infer':
        assert os.path.isdir(cfg['load_path'])
        model_cfg = Configs.load_cfg_from_file(Configs.gen_cfg_path(cfg['load_path']))
        assert model_cfg is not None
        infer(model_cfg, cfg)
    elif cfg['mode'] == 'decoding':
        assert os.path.isdir(cfg['load_path'])
        model_cfg = Configs.load_cfg_from_file(Configs.gen_cfg_path(cfg['load_path']))
        assert model_cfg is not None
        decoding(model_cfg, cfg)
    elif cfg['mode'] == 'parallel_test':
        assert os.path.isdir(cfg['load_path'])
        model_cfg = Configs.load_cfg_from_file(Configs.gen_cfg_path(cfg['load_path']))
        assert model_cfg is not None
        parallel_test(model_cfg, cfg)


if __name__ == '__main__':
    tf.app.run()
