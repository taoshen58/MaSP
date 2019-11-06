import os
import logging
import platform
import argparse
import time
import pickle
from glob import glob

from os.path import join
from tensorflow.contrib.training import HParams
from abc import ABCMeta, abstractmethod
from collections import defaultdict

from peach.utils.string import underline_to_camel, mp_join


def merge_hparams(p1, p2):
    params = HParams()
    v1 = p1.values()
    v2 = p2.values()
    for (k, v) in v1.items():
        params.add_hparam(k, v)
    for (k, v) in v2.items():
        params.add_hparam(k, v)
    return params


def print_hparams(params, sort=True, print_std=True):
    kv_list = [(k, v) for (k, v) in params.values().items()]
    if sort:
        kv_list = list(sorted(
            kv_list,
            key=lambda elem: elem[0]
        ))
    str_re = ''
    for (k, v) in kv_list:
        str_re += "%s: %s%s" % (k, v, os.linesep)
    if print_std:
        logging.info(str_re)
    return str_re


class HParamsCenter(object):
    default_namespace='other_hparams'

    def __init__(self, default_preprocessing_hparams=None, default_model_hparams=None, default_training_hparams=None,
                 models_dir="src.models"):
        self.hparams_dict = defaultdict(HParams)
        self.models_dir = models_dir

        # parsed
        args = self._parsed_args()
        self.parsed_hparams = HParams()
        for key, val in args.__dict__.items():
            self.parsed_hparams.add_hparam(key, val)
        self.register_hparams(self.parsed_hparams, 'parsed_hparams')

        # pre-processing
        self.preprocessing_hparams = default_preprocessing_hparams or HParams()
        self.preprocessing_hparams.parse(self.parsed_hparams.preprocessing_hparams)
        self.register_hparams(self.preprocessing_hparams, 'preprocessing_hparams')

        # model
        self.model_hparams = default_model_hparams or HParams()
        self.model_hparams = merge_hparams(
            self.model_hparams,
            self._fetch_model_specific_hparams(self.parsed_hparams.network_class, self.parsed_hparams.network_type, self.models_dir))
        self.model_hparams.parse(self.parsed_hparams.model_hparams)
        self.register_hparams(self.model_hparams, 'model_hparams')

        # traning
        self.training_hparams = default_training_hparams or HParams()
        self.training_hparams.parse(self.parsed_hparams.training_hparams)
        self.register_hparams(self.training_hparams, 'training_hparams')

    @staticmethod
    def _fetch_model_specific_hparams(network_class, network_type, models_dir):
        model_hparams = HParams(
            model_class=None
        )
        if network_class is not None and network_type is not None:
            model_module_name = 'model_%s' % network_type
            model_class_name = underline_to_camel(model_module_name)
            try:
                src_module = __import__('%s.%s.%s' % (models_dir, network_class, model_module_name))
                model_class = eval('src_module.models.%s.%s.%s' % (network_class, model_module_name, model_class_name))
                model_hparams = model_class.get_default_model_parameters()
                model_hparams.add_hparam('model_class', model_class)  # add model class
            except ImportError:
                print('Fatal Error: no model module: \"src.models.%s.%s\"' % (network_class, model_module_name))
            except AttributeError:
                print('Fatal Error: probably (1) no model class named as %s.%s, '
                      'or (2) the class no \"get_default_model_parameters()\"' % (network_class, model_module_name))
        return model_hparams

    @staticmethod
    def _parsed_args():
        parser = argparse.ArgumentParser()
        parser.register('type', 'bool', (lambda x: x.lower() in ("yes", "true", "t", "1")))
        parser.add_argument('--mode', type=str, default='train', help='')
        parser.add_argument('--dataset', type=str, default='none', help='')
        parser.add_argument('--network_class', type=str, default='transformer', help='')
        parser.add_argument('--network_type', type=str, default=None, help='')
        parser.add_argument('--gpu', type=str, default='3', help='selected gpu index')
        parser.add_argument('--gpu_mem', type=float, default=None, help='selected gpu index')
        parser.add_argument('--model_dir_prefix', type=str, default='prefix', help='model dir name prefix')
        parser.add_argument('--machine', type=str, default='none', help='using aws')

        # parsing parameters group
        parser.add_argument('--preprocessing_hparams', type=str, default='', help='')
        parser.add_argument('--model_hparams', type=str, default='', help='')
        parser.add_argument('--training_hparams', type=str, default='', help='')

        parser.set_defaults(shuffle=True)
        return parser.parse_args()

    def register_hparams(self, hparams, name):
        assert isinstance(hparams, HParams)
        assert isinstance(name, str)

        if name in self.hparams_dict:
            self.hparams_dict[name] = merge_hparams(self.hparams_dict[name], hparams)
        else:
            self.hparams_dict[name] = hparams

    @property
    def all_hparams(self):
        all_hparams = HParams()
        for name, hp in self.hparams_dict.items():
            all_hparams = merge_hparams(all_hparams, hp)
        return all_hparams

    def __setitem__(self, key, value):
        assert isinstance(key, str)
        # this is added to the default
        # self.hparams_dict[self.default_namespace][key] = value
        key_found = False
        for _, hps in self.hparams_dict.items():
            try:
                if key in hps:
                    key_found = True
            # when tf==1.4.1, directly use "in" will raise TypeError: argument of type 'HParams' is not iterable
            except TypeError:
                if key in hps.values():
                    key_found = True
            if key_found:
                hps.set_hparam(key, value)
                break

        if not key_found:  # not found, set it
            self.hparams_dict[self.default_namespace].add_hparam(key, value)

    def __getitem__(self, item):
        assert isinstance(item, str)

        for name, hp in self.hparams_dict.items():
            try:
                return getattr(hp, item)
            except AttributeError:
                pass
        raise AttributeError('no item named as \'%s\'' % item)

    def __contains__(self, item):
        if isinstance(item, str):
            for key, hps in self.hparams_dict.items():
                try:
                    if item in hps:
                        return True
                # when tf==1.4.1, directly use "in" will raise TypeError: argument of type 'HParams' is not iterable
                except TypeError:
                    if item in hps.values():
                        return True
        return False


class ConfigsTemplate(metaclass=ABCMeta):
    def __init__(self, hparams_center, project_name):
        # add default and parsed parameters to cfg
        self.hparams_center = hparams_center
        self.project_name = project_name
        self.dataset_dir, self.project_dir = self._project_dataset_dirs()

        self.processed_name, self.model_name, self.ckpt_name, self.log_name, \
            self.raw_data_dir, self.train_data_name, self.dev_data_name, self.test_data_name = self._file_names()

        self._file_dirs_paths()
        self._add_to_hparams_center()
        self._other_setup()

    def save_cfg_to_file(self, file_path):
        logging.info("save cfg to {}".format(file_path))
        with open(file_path, "wb") as fp:
            pickle.dump(self, fp)

    @classmethod
    def load_cfg_from_file(cls, file_path):
        logging.info("load cfg from {}".format(file_path))
        try:
            with open(file_path, "rb") as fp:
                loaded = pickle.load(fp)
                logging.info("\tsuccess to load")
                return loaded
        except FileNotFoundError:
            logging.info("\tfailure to load")
        return None

    def logging_hparams(self):
        logging.info(print_hparams(self.all_hparams, print_std=False))

    def _project_dataset_dirs(self):
        self.runtime_system = platform.system()
        if self.runtime_system == 'Darwin':
            dataset_dir = 'data'
            project_dir = './'
        elif self.runtime_system == 'Linux':
            if self.hparams_center['machine'] == 'aws':
                dataset_dir = 'data'
                project_dir = './'
            elif platform.platform().find('redhat') >= 0:
                dataset_dir = 'data'
                project_dir = './'
            else:
                dataset_dir = 'data'
                project_dir = './'
        else:
            raise (SystemError, 'Have not found the configs corresponding to current system')
        return dataset_dir, project_dir

    @abstractmethod
    def _file_names(self):
        processed_name = self.get_params_str(
            ['dataset']
        ) + '_proprec.pickle'

        if self['network_type'] is None or self['network_type'] == 'test':
            model_name = '_test'
        else:
            model_name_params = ['dataset', 'network_class', 'network_type']
            if self['model_class'] is not None:
                model_name_params += self['model_class'].get_identity_param_list()
            else:
                print('fatal error: can not reach the model class')
            model_name = self.get_params_str(model_name_params)

        ckpt_name = 'model_file.ckpt'
        log_name = 'log_' + ConfigsTemplate.time_suffix() + '.txt'

        # Add dataset: from self['dataset']
        raw_data_dir, train_data_name, dev_data_name, test_data_name = None, None, None, None

        return processed_name, model_name, ckpt_name, log_name, \
               raw_data_dir, train_data_name, dev_data_name, test_data_name

    def _file_dirs_paths(self):
        # -------  dir -------
        self.bpe_data_dir = join(self.dataset_dir, 'bpe')
        self.pretrained_transformer_dir = join(self.dataset_dir, 'pretrained_transformer')
        self.glove_dir = join(self.dataset_dir, 'glove')

        self.runtime_dir = mp_join(self.project_dir, 'runtime')
        self.run_model_dir = mp_join(self.runtime_dir, 'run_model')
        self.processed_dir = mp_join(self.runtime_dir, 'preproc')

        self.runtime_dir = mp_join(self.project_dir, 'runtime')
        self.run_model_dir = mp_join(self.runtime_dir, 'run_model')
        self.processed_dir = mp_join(self.runtime_dir, 'preproc')

        self.cur_run_dir = mp_join(self.run_model_dir, self['model_dir_prefix'] + self.model_name)
        self.log_dir = mp_join(self.cur_run_dir, 'log_files')
        self.summary_dir = mp_join(self.cur_run_dir, 'summary')
        self.ckpt_dir = mp_join(self.cur_run_dir, 'ckpt')
        self.other_dir = mp_join(self.cur_run_dir, 'other')

        # path
        self.train_data_path = join(self.raw_data_dir, self.train_data_name)
        self.dev_data_path = join(self.raw_data_dir, self.dev_data_name)
        self.test_data_path = join(self.raw_data_dir, self.test_data_name)

        self.processed_path = join(self.processed_dir, self.processed_name)
        self.ckpt_path = join(self.ckpt_dir, self.ckpt_name)
        self.log_path = join(self.log_dir, self.log_name)

        self.cfg_path = join(self.ckpt_dir, 'cfg.pkl')

    def _add_to_hparams_center(self):
        path_hparams = HParams(
            dataset_dir=self.dataset_dir,
            project_dir=self.project_dir,
            # dir
            bpe_data_dir=self.bpe_data_dir,
            pretrained_transformer_dir=self.pretrained_transformer_dir,
            glove_dir=self.glove_dir,
            runtime_dir=self.runtime_dir,
            run_model_dir=self.run_model_dir,
            processed_dir=self.processed_dir,
            cur_run_dir=self.cur_run_dir,
            log_dir=self.log_dir,
            summary_dir=self.summary_dir,
            ckpt_dir=self.ckpt_dir,
            other_dir=self.other_dir,

            # path
            raw_data_dir=self.raw_data_dir,
            train_data_path=self.train_data_path,
            dev_data_path=self.dev_data_path,
            test_data_path=self.test_data_path,

            processed_path=self.processed_path,
            ckpt_path=self.ckpt_path,
            log_path=self.log_path,
            cfg_path=self.cfg_path,
        )
        self.hparams_center.register_hparams(path_hparams, 'path_hparams')

    def _other_setup(self):
        logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO, datefmt='%m/%d %I:%M:%S %p')
        file_handler = logging.FileHandler(self.log_path)  # add a file handler to a logger
        logging.getLogger().addHandler(file_handler)
        # cuda support
        os.environ['CUDA_VISIBLE_DEVICES'] = '' if self['gpu'].lower() == 'none' else self['gpu']

        self.hparams_center['intX'] = 'int32'
        self.hparams_center['floatX'] = 'float32'

    def get_params_str(self, params):
        assert self.hparams_center is not None

        def abbreviation(name):
            words = name.strip().split('_')
            abb = ''
            for word in words:
                abb += word[0]
            return abb

        abbreviations = map(abbreviation, params)
        model_params_str = ''
        for paramsStr, abb in zip(params, abbreviations):
            model_params_str += '_' + abb + '.' + str(eval('self[\'%s\']' % paramsStr))
        return model_params_str

    @staticmethod
    def gen_cfg_path(ckpt_dir):
        cfg_file_pattern = os.path.join(ckpt_dir, "cfg*pkl")
        cfg_file_path = list(glob(cfg_file_pattern))
        assert len(cfg_file_path) == 1, \
            "with cfg_file_pattern cfg_file_path\'{}\' find {} candidates pkl file".format(cfg_file_pattern, len(cfg_file_path))
        cfg_file_path = cfg_file_path[0]  # remove the list
        return cfg_file_path

    @staticmethod
    def time_suffix():
        return '-'.join(time.asctime(time.localtime(time.time())).split()[1:-1]).replace(':', '-')

    @property
    def all_hparams(self):
        return self.hparams_center.all_hparams

    def __getitem__(self, item):
        return self.hparams_center[item]

    def __setitem__(self, key, value):
        self.hparams_center[key] = value

    def __contains__(self, item):
        return (item in self.hparams_center)










