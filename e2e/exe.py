from tqdm import tqdm
import random
from BFS.parser import Base_action, Parser
from BFS.agent import KB
from copy import deepcopy
import timeout_decorator
import time


class SketchExecutor(object):
    def __init__(self):
        self._leaf_args = set(["e", "r", "Type", "num_utterence"])
        self._actions = set()
        self._dict_action_in = {}
        self._dict_action_out = {}
        self._dict_out_action = {}
        for act in Base_action:
            self._actions.add(act)
            in_tuple = Base_action[act][1]
            if isinstance(in_tuple, tuple):
                assert len(in_tuple) >= 2
                in_tuple = in_tuple[1:]
            else:
                assert isinstance(in_tuple, str)
                in_tuple = (in_tuple,)
            self._dict_action_in[act] = tuple(in_tuple)
            self._dict_action_out[act] = Base_action[act][0]

            if Base_action[act][0] not in self._dict_out_action:
                self._dict_out_action[Base_action[act][0]] = []
            self._dict_out_action[Base_action[act][0]].append(act)
        # other for _dict_out_action
        for _arg in self._leaf_args:
            self._dict_out_action[_arg] = [_arg]

        self._all_args = set()
        for _args in self._dict_action_in.values():
            self._all_args.update(_args)

    @property
    def valid_action_set(self):
        return self._actions

    @property
    def valid_arg_set(self):
        return self._all_args

    @property
    def action2inargs_dict(self):
        return self._dict_action_in

    @property
    def action2outtype_dict(self):
        return self._dict_action_out

    @property
    def outtype2actionlist_dict(self):
        return self._dict_out_action

    def sketch_exe(self, sketch_input):
        def _success_cond(_sketch):
            if len(_sketch) == 1 and _sketch[0] =="S":
                return True

        def _pre_process(_sketch):
            # insert "Count" for "A26"
            _len = len(_sketch)
            for _idx in range(_len-1, -1, -1):
                if _sketch[_idx] == "A26":
                    _sketch.insert(_idx+1, "Count")
            return _sketch

        def _post_process(_sketch):
            _sketch = list(_sketch)
            # remove all "Count"
            _len = len(_sketch)
            for _idx in range(_len - 1, -1, -1):
                if _sketch[_idx] == "Count":
                    _sketch.pop(_idx)
            return _sketch

        sketch_input = list(sketch_input)
        sketch_input = _pre_process(sketch_input)

        # sanity check: the wrong input
        for _token in sketch_input:
            if not (_token in self._all_args or _token in self._actions):
                return False, _post_process(sketch_input)
        if len(sketch_input) == 0:
            return False, list()

        sketch_res = sketch_input
        # grammar check 1: early stage
        if _success_cond(sketch_res):
            return True, _post_process(sketch_res)
        elif not sketch_res[0] in ["A1", "A2", "A3"]:
            return False, _post_process(sketch_res)
        action_stack = []
        args_stack = []
        tmp_args = []
        for _idx_t, _token in enumerate(sketch_res):
            if _token in self._actions:
                if len(action_stack) > 0:  # check the arg input
                    args_stack.append(tmp_args)
                    tmp_args = []
                action_stack.append(_token)
            else:
                tmp_args.append(_token)
        args_stack.append(tmp_args)

        # start to execute
        exe_status = "normal"  # "error"

        num_act = len(action_stack)
        answer_list = []
        for _idx_a in range(num_act-1, -1, -1):
            cur_act = action_stack[_idx_a]
            answer_list = args_stack[_idx_a] + answer_list
            # if cur_act == "A26":
            #     answer_list.insert(0, "Count")
            tgt_args = self._dict_action_in[cur_act]

            tgt_len = len(tgt_args)
            if tgt_args == tuple(answer_list[:tgt_len]):
                out_type = self._dict_action_out[cur_act]
                answer_list = [out_type] + answer_list[tgt_len:]
            else:
                # check the grammar
                _tmp_args = []
                for _idx_s in range(len(tgt_args)):
                    if _idx_s >= len(answer_list):
                        break
                    if answer_list[_idx_s] in self._actions:
                        _tmp_args.append(self._dict_action_out[answer_list[_idx_s]])
                        break
                    else:
                        _tmp_args.append(answer_list[_idx_s])
                if tuple(_tmp_args) != tuple(self._dict_action_in[cur_act][:len(_tmp_args)]):
                    exe_status = "grammar_error"
                    break
                answer_list.insert(0, cur_act)
        if exe_status == "grammar_error":
            return False, list()

        if _success_cond(answer_list):
            return True, _post_process(answer_list)
        else:
            return None, _post_process(answer_list)

    def sketch_exe_v2(self, sketch_input):
        def _success_cond(_sketch):
            if len(_sketch) == 1 and _sketch[0] =="S":
                return True

        def _pre_process(_sketch):
            # insert "Count" for "A26"
            _sketch = list(_sketch)
            _len = len(_sketch)
            for _idx in range(_len-1, -1, -1):
                if _sketch[_idx] == "A26":
                    _sketch.insert(_idx+1, "Count")
            return _sketch

        def _post_process(_sketch):
            if _sketch is None:
                return None
            _sketch = list(_sketch)
            # remove all "Count"
            _len = len(_sketch)
            for _idx in range(_len - 1, -1, -1):
                if _sketch[_idx] == "Count":
                    _sketch.pop(_idx)
            return _sketch

        # pre process sketch
        sketch_input = _pre_process(sketch_input)

        # sanity check: the wrong input
        for _token in sketch_input:
            if not (_token in self._all_args or _token in self._actions):
                return False, list()
        if len(sketch_input) == 0:
            return False, list()

        # grammar check 1: early stage
        if _success_cond(sketch_input):
            return True, _post_process(sketch_input)
        elif not sketch_input[0] in ["A1", "A2", "A3"]:
            return False, _post_process(sketch_input)

        len_sketch = len(sketch_input)
        answer_list = []
        exe_status = "normal"  # or "error"
        for _idx_t in range(len_sketch-1, -1, -1):
            _token = sketch_input[_idx_t]
            if _token in self._actions:
                cur_act = _token
                tgt_args = self._dict_action_in[cur_act]
                tgt_len = len(tgt_args)
                if tgt_args == tuple(answer_list[:tgt_len]):
                    out_type = self._dict_action_out[cur_act]
                    answer_list = [out_type] + answer_list[tgt_len:]
                else:
                    # check the grammar
                    _tmp_args = []
                    for _idx_s in range(len(tgt_args)):
                        if _idx_s >= len(answer_list):
                            break
                        if answer_list[_idx_s] in self._actions:
                            _tmp_args.append(self._dict_action_out[answer_list[_idx_s]])
                            break
                        else:
                            _tmp_args.append(answer_list[_idx_s])
                    if tuple(_tmp_args) != tuple(self._dict_action_in[cur_act][:len(_tmp_args)]):
                        exe_status = "grammar_error"
                        break
                    answer_list.insert(0, _token)

            else:
                answer_list.insert(0, _token)
        if exe_status == "grammar_error":
            return False, list()

        answer_list = _post_process(answer_list)
        if _success_cond(answer_list):
            return True, answer_list
        else:
            return None, answer_list

    def sketch_exe_with_next(self, sketch_input):
        if len(sketch_input) == 0 or (
                len(sketch_input) == 1 and (sketch_input[0] not in self._actions
                                            and sketch_input[0] not in self._all_args)):
            return None, list(), self._dict_out_action["S"]

        is_succ, run_res = self.sketch_exe_v2(sketch_input)

        if is_succ is None:
            cur_args = []
            cur_action = None
            for _idx_t in range(len(run_res)-1, -1, -1):

                if run_res[_idx_t] in self._actions:
                    cur_action = run_res[_idx_t]
                    break
                else:
                    cur_args.insert(0, run_res[_idx_t])
            assert cur_action is not None
            tgt_args = self._dict_action_in[cur_action]
            if cur_action == "A26":  # remove the "Counter"
                tgt_args = tgt_args[1:]
            assert len(cur_args) < len(tgt_args)
            next_arg = tgt_args[len(cur_args)]
            return None, run_res, self._dict_out_action[next_arg]
        else:
            return is_succ, run_res, None


class LfExecutor(object):
    def __init__(self, sketch_executor=None, kb_mode="offline", use_op_type_constraint=False):
        self.skt_exe = sketch_executor or SketchExecutor()
        kb = KB(kb_mode)
        self._parser = Parser(kb, use_op_type_constraint=use_op_type_constraint)
        self._parser.load_child2parent()
        self._actions = set()
        for act in Base_action:
            self._actions.add(act)
        self._action_set_interact = set(["A7", "A8", "A9"])

    @timeout_decorator.timeout(1.2)
    def arg_set_lf_exe(self, slf_input):
        # the only different between lf_input and slf_input is slf have extra flag
        def _success_cond(_lf):
            if len(_lf) == 1 and _lf[0][0] == "S" and _lf[0][1] is not None:
                return True

        def _pre_process(_lf):
            # insert "Count" for "A26"
            _lf = list(_lf)
            _len = len(_lf)
            for _idx in range(_len-1, -1, -1):
                if _lf[_idx][1] == "A26":
                    if (_idx == _len-1) or _lf[_idx+1][0] != "Count":
                        _lf.insert(_idx+1, ("Count", "Count", False))
            return _lf

        def _post_process(_lf):
            if _lf is None:
                return None
            _lf = list(_lf)
            # remove all "Count"
            _len = len(_lf)
            for _idx in range(_len - 1, -1, -1):
                if _lf[_idx][0] == "Count":
                    _lf.pop(_idx)
            return _lf

        slf_input = _pre_process(slf_input)

        if len(slf_input) == 0:
            return False, list()
        for _type, _val, _flag in slf_input:
            if _type == "Action":
                assert not _flag
                if _val not in self.skt_exe.valid_action_set:
                    return False, list()
            else:
                if _flag: # all actions' flag is False
                    assert isinstance(_val, list)
                if _type not in self.skt_exe.valid_arg_set or _val is None:
                    return False, list()

        # grammar check 1: early stage
        if _success_cond(slf_input):
            return True, _post_process(slf_input)
        elif slf_input[0][1] not in ["A1", "A2", "A3"]:
            return False, _post_process(slf_input)

        len_lf = len(slf_input)
        ans_type_list = []
        ans_val_list = []
        ans_flag_list = []
        exe_status = "normal"  # or "error"
        for _idx_t in range(len_lf - 1, -1, -1):
            _type, _val, _flag = slf_input[_idx_t]
            if _type != "Action":
                ans_type_list.insert(0, _type)
                ans_val_list.insert(0, _val)
                ans_flag_list.insert(0, _flag)
            else:
                cur_act = _val
                tgt_args = tuple(self.skt_exe.action2inargs_dict[cur_act])
                tgt_len = len(tgt_args)
                if tgt_args == tuple(ans_type_list[:tgt_len]):
                    before_inp_vals = ans_val_list[:tgt_len]
                    before_inp_flags = ans_flag_list[:tgt_len]
                    # all possible vals
                    all_inp_vals = [[]]
                    for _inp_val, _inp_flag in zip(before_inp_vals, before_inp_flags):
                        if not _inp_flag:
                            for _elem in all_inp_vals:
                                _elem.append(_inp_val)
                        else:
                            assert isinstance(_inp_val, list)
                            new_all_inp_vals = []
                            for _sub_inp_val in _inp_val:
                                copied_all_inp_vals = deepcopy(all_inp_vals)
                                for _elem in copied_all_inp_vals:
                                    _elem.append(_sub_inp_val)
                                new_all_inp_vals.extend(copied_all_inp_vals)
                            all_inp_vals = new_all_inp_vals

                    out_type = self.skt_exe.action2outtype_dict[cur_act]
                    if isinstance(Base_action[cur_act][1], tuple):
                        all_out_vals = []
                        for _inp_vals in all_inp_vals:
                            assert len(_inp_vals) == tgt_len
                            out_val = self._parser.op(Base_action[cur_act][1][0], _inp_vals)
                            if out_val is not None and out_val not in all_out_vals:
                                all_out_vals.append(out_val)
                    else:
                        assert tgt_len == 1
                        all_out_vals = []
                        for _inp_vals in all_inp_vals:
                            assert len(_inp_vals) == 1 and _inp_vals[0] is not None
                            all_out_vals.append(_inp_vals[0])

                    if len(all_out_vals) == 0:
                        exe_status = "run_error"
                        break
                    elif len(all_out_vals) == 1:
                        all_out_vals = all_out_vals[0]
                        out_flag = False
                    else:
                        out_flag = True

                    ans_type_list = [out_type] + ans_type_list[tgt_len:]
                    ans_val_list = [all_out_vals] + ans_val_list[tgt_len:]
                    ans_flag_list = [out_flag] + ans_flag_list[tgt_len:]
                else:
                    # check the grammar
                    _tmp_args = []
                    for _idx_s in range(tgt_len):
                        if _idx_s >= len(ans_type_list):
                            break
                        if ans_type_list[_idx_s] == "Action":
                            _tmp_args.append(self.skt_exe.action2outtype_dict[ans_val_list[_idx_s]])
                            break
                        else:
                            _tmp_args.append(ans_type_list[_idx_s])
                    if tuple(_tmp_args) != tuple(tgt_args[:len(_tmp_args)]):
                        exe_status = "grammar_error"
                        break
                    # insert the action
                    ans_type_list.insert(0, _type)
                    ans_val_list.insert(0, _val)
                    ans_flag_list.insert(0, _flag)
        if exe_status == "run_error":
            return False, list()
        elif exe_status == "grammar_error":
            return False, list()

        ans_lf = _post_process(list(zip(ans_type_list, ans_val_list, ans_flag_list)))

        if _success_cond(ans_lf):
            return True, ans_lf
        else:
            return None, ans_lf

    def arg_set_lf_exe_with_next(self, slf_input):
        if len(slf_input) == 0 or (  # the input is empty lf
            len(slf_input) == 1 and (slf_input[0][0] != "Action" and
                                     slf_input[0][1] not in self.skt_exe.valid_arg_set)
        ):
            return None, list(), self.skt_exe.outtype2actionlist_dict["S"]

        try:
            is_succ, ans_lf = self.arg_set_lf_exe(slf_input)
        except timeout_decorator.TimeoutError:
            return False, list(), None

        if is_succ is None:
            ans_sketch = [_val if _type == "Action" else _type for _type, _val, _flag in ans_lf]
            cur_args = []
            cur_action = None
            for _idx_t in range(len(ans_sketch) - 1, -1, -1):
                if ans_sketch[_idx_t] in self._actions:
                    cur_action = ans_sketch[_idx_t]
                    break
                else:
                    cur_args.insert(0, ans_sketch[_idx_t])
            assert cur_action is not None
            tgt_args = self.skt_exe.action2inargs_dict[cur_action]
            if cur_action == "A26":  # remove the "Count"
                tgt_args = tgt_args[1:]
            assert len(cur_args) < len(tgt_args)
            next_arg = tgt_args[len(cur_args)]
            return None, ans_lf, self.skt_exe.outtype2actionlist_dict[next_arg]
        else:
            return is_succ, ans_lf, None

    # @timeout_decorator.timeout(1)
    # def lf_exe(self, lf_input):  # the logical form is a list of (Type, LF_TOKEN)
    #     def _success_cond(_lf):
    #         if len(_lf) == 1 and _lf[0][0] == "S" and _lf[0][1] is not None:
    #             return True
    #
    #     def _pre_process(_lf):
    #         # insert "Count" for "A26"
    #         _lf = list(_lf)
    #         _len = len(_lf)
    #         for _idx in range(_len-1, -1, -1):
    #             if _lf[_idx][1] == "A26":
    #                 if (_idx == _len-1) or _lf[_idx+1][0] != "Count":
    #                     _lf.insert(_idx+1, ("Count", "Count"))
    #         return _lf
    #
    #     def _post_process(_lf):
    #         if _lf is None:
    #             return None
    #         _lf = list(_lf)
    #         # remove all "Count"
    #         _len = len(_lf)
    #         for _idx in range(_len - 1, -1, -1):
    #             if _lf[_idx][0] == "Count":
    #                 _lf.pop(_idx)
    #         return _lf
    #
    #     lf_input = _pre_process(lf_input)
    #
    #     # sanity check: the wrong input
    #     if len(lf_input) == 0:
    #         return False, list()
    #     for _type, _val in lf_input:
    #         if _type == "Action":
    #             if _val not in self.skt_exe.valid_action_set:
    #                 return False, list()
    #         else:
    #             if _type not in self.skt_exe.valid_arg_set or _val is None:
    #                 return False, list()
    #
    #     # grammar check 1: early stage
    #     if _success_cond(lf_input):
    #         return True, _post_process(lf_input)
    #     elif lf_input[0][1] not in ["A1", "A2", "A3"]:
    #         return False, _post_process(lf_input)
    #
    #     len_lf = len(lf_input)
    #     ans_type_list = []
    #     ans_val_list = []
    #     exe_status = "normal"  # or "error"
    #     for _idx_t in range(len_lf - 1, -1, -1):
    #         _type, _val = lf_input[_idx_t]
    #         if _type == "Action":
    #             cur_act = _val
    #             tgt_args = self.skt_exe.action2inargs_dict[cur_act]
    #             tgt_len = len(tgt_args)
    #             if tgt_args == tuple(ans_type_list[:tgt_len]):
    #                 inp_vals = ans_val_list[:tgt_len]
    #                 # run op
    #                 out_type = self.skt_exe.action2outtype_dict[cur_act]
    #                 if isinstance(Base_action[cur_act][1], tuple):
    #                     out_val = self._parser.op(Base_action[cur_act][1][0], inp_vals)
    #                     if out_val is None:
    #                         exe_status = "run_error"
    #                         break
    #                 else:
    #                     assert tgt_len == 1
    #                     out_val = inp_vals[0]
    #                 ans_type_list = [out_type] + ans_type_list[tgt_len:]
    #                 ans_val_list = [out_val] + ans_val_list[tgt_len:]
    #             else:
    #                 # check the grammar
    #                 _tmp_args = []
    #                 for _idx_s in range(tgt_len):
    #                     if _idx_s >= len(ans_type_list):
    #                         break
    #                     if ans_type_list[_idx_s] == "Action":
    #                         _tmp_args.append(self.skt_exe.action2outtype_dict[ans_val_list[_idx_s]])
    #                         break
    #                     else:
    #                         _tmp_args.append(ans_type_list[_idx_s])
    #                 if tuple(_tmp_args) != tuple(tgt_args[:len(_tmp_args)]):
    #                     exe_status = "grammar_error"
    #                     break
    #                 # insert the action
    #                 ans_type_list.insert(0, _type)
    #                 ans_val_list.insert(0, _val)
    #         else:
    #             ans_type_list.insert(0, _type)
    #             ans_val_list.insert(0, _val)
    #
    #     if exe_status == "grammar_error":
    #         return False, list()
    #     elif exe_status == "run_error":
    #         return False, list()
    #
    #     ans_lf = _post_process(list(zip(ans_type_list, ans_val_list)))
    #
    #     if _success_cond(ans_lf):
    #         return True, ans_lf
    #     else:
    #         return None, ans_lf
    #
    # def lf_exe_with_next(self, lf_input):
    #     if len(lf_input) == 0 or (  # the input is empty lf
    #         len(lf_input) == 1 and (lf_input[0][0] != "Action" and
    #                                 lf_input[0][1] not in self.skt_exe.valid_arg_set)
    #     ):
    #         return None, list(), self.skt_exe.outtype2actionlist_dict["S"]
    #
    #     is_succ, ans_lf = self.lf_exe(lf_input)
    #
    #     if is_succ is None:
    #         ans_sketch = [_val if _type == "Action" else _type for _type, _val in ans_lf]
    #         cur_args = []
    #         cur_action = None
    #         for _idx_t in range(len(ans_sketch) - 1, -1, -1):
    #             if ans_sketch[_idx_t] in self._actions:
    #                 cur_action = ans_sketch[_idx_t]
    #                 break
    #             else:
    #                 cur_args.insert(0, ans_sketch[_idx_t])
    #         assert cur_action is not None
    #         tgt_args = self.skt_exe.action2inargs_dict[cur_action]
    #         if cur_action == "A26":  # remove the "Count"
    #             tgt_args = tgt_args[1:]
    #         assert len(cur_args) < len(tgt_args)
    #         next_arg = tgt_args[len(cur_args)]
    #         return None, ans_lf, self.skt_exe.outtype2actionlist_dict[next_arg]
    #     else:
    #         return is_succ, ans_lf, None
