import tensorflow as tf
import os


class PerformanceRecoder(object):
    def __init__(self, ckpt_dir, save_model, num_top=3):
        self.ckpt_dir = ckpt_dir
        self.save_model = save_model
        self.num_top = num_top
        self.top_list = []
        self.saver = tf.train.Saver(max_to_keep=None)

    def update_top_list(self, global_step, dev_accu, sess):
        cur_ckpt_path = self._ckpt_file_path_generator(global_step)
        self.top_list.append([global_step, dev_accu])
        self.top_list = list(sorted(self.top_list, key=lambda elem: elem[1], reverse=True))

        if len(self.top_list) <= self.num_top:
            self._create_ckpt_file(sess, cur_ckpt_path)
            return True, None
        elif len(self.top_list) == self.num_top + 1:
            out_state = self.top_list[-1]
            self.top_list = self.top_list[:-1]
            if out_state[0] == global_step:
                return False, None
            else:  # add and delete
                self._delete_ckpt_file(self._ckpt_file_path_generator(out_state[0]))
                self._create_ckpt_file(sess, cur_ckpt_path)
                return True, out_state[0]
        else:
            raise RuntimeError()

    def get_best(self):
        try:
            return self.top_list[0]
        except IndexError:
            return None, None

    def get_best_str(self):
        try:
            return 'best: step-%d, val-%f' % (self.top_list[0][0], self.top_list[0][1])
        except IndexError:
            return 'best: step-None, val-None'

    def _update_ckpt_file(self):
        if len(self.top_list) > 0:
            with open(os.path.join(self.ckpt_dir, "checkpoint"), "w", encoding="utf-8") as fp:
                fp.write("model_checkpoint_path: \"{}\"".format(os.path.basename(self._ckpt_file_path_generator(self.top_list[0][0]))))
                fp.write(os.linesep)
                for step, _ in self.top_list:
                    fp.write("all_model_checkpoint_paths: \"{}\"".format(os.path.basename(self._ckpt_file_path_generator(step))))
                    fp.write(os.linesep)

    def _ckpt_file_path_generator(self, step):
        return os.path.join(self.ckpt_dir, 'top_result_saver_step_%d.ckpt' % step)

    def _delete_ckpt_file(self, ckpt_file_path):
        if os.path.isfile(ckpt_file_path+'.meta'):
            os.remove(ckpt_file_path+'.meta')
        if os.path.isfile(ckpt_file_path+'.index'):
            os.remove(ckpt_file_path+'.index')
        if os.path.isfile(ckpt_file_path+'.data-00000-of-00001'):
            os.remove(ckpt_file_path+'.data-00000-of-00001')

    def _create_ckpt_file(self, sess, ckpt_file_path):
        if self.save_model:
            self.saver.save(sess, ckpt_file_path)
        self._update_ckpt_file()
