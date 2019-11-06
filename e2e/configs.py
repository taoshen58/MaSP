from peach.utils.hparams import HParams, HParamsCenter, ConfigsTemplate
from os.path import join
import os


class Configs(ConfigsTemplate):
    def __init__(self, hparams_center, project_name):
        super(Configs, self).__init__(hparams_center, project_name)

        self['dev_list_file'] = os.path.join(self["processed_dir"], "dev_list_file.txt")

        if 'bert_pretrained_dir' in self:
            self['vocab_file'] = join(self['bert_pretrained_dir'], "vocab.txt")
            self['init_checkpoint'] = join(self['bert_pretrained_dir'], "bert_model.ckpt")
            self['bert_config_file'] = join(self['bert_pretrained_dir'], "bert_config.json")
            if os.path.basename(self['bert_pretrained_dir'].strip().strip("/")):
                self['do_lower_case'] = True
            else:
                self['do_lower_case'] = False
        else:
            self['vocab_file'] = 'none'
            self['init_checkpoint'] = 'none'
            self['bert_config_file'] = 'none'
            self['do_lower_case'] = True

        self.logging_hparams()

    def _file_names(self):
        processed_name = self.get_params_str(
            ['dataset', ]
        ) + '_proprec.pickle'

        if self['network_type'] is None or self['network_type'] == 'test':
            model_name = '_test'
        else:
            model_name_params = ['dataset', 'network_class', 'network_type', 'max_sequence_len']
            if self['model_class'] is not None:
                model_name_params += self['model_class'].get_identity_param_list()
            else:
                print('fatal error: can not reach the model class')
            model_name = self.get_params_str(model_name_params)

        ckpt_name = 'model_file.ckpt'
        log_name = 'log_' + ConfigsTemplate.time_suffix() + '.txt'

        raw_data_dir = join(self.project_dir, "data/BFS")
        if self["dataset"] == "e2e_wo_con":
            train_data_name, dev_data_name = 'train_proc_direct_1000_wo_con', 'dev_proc_direct_1000_subset_wo_con'
        else:
            raise AttributeError

        test_data_name = "test"

        return processed_name, model_name, ckpt_name, log_name, raw_data_dir, \
               train_data_name, dev_data_name, test_data_name
