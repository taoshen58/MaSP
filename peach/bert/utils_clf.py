import tensorflow as tf
import csv
import os
import math
from peach.bert import tokenization
from peach.utils.tree.shift_reduce import shift_reduce_constituency_forest


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, segment_ids, label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id


# ===================================#
# ===================================#
class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @property
  def no_label_for_test(self):
    return True

  @property
  def is_paired_data(self):
    return True

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(
        os.path.join(data_dir, "multinli",
                     "multinli.train.%s.tsv" % self.language))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      if label == tokenization.convert_to_unicode("contradictory"):
        label = tokenization.convert_to_unicode("contradiction")
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % (i)
      language = tokenization.convert_to_unicode(line[0])
      if language != tokenization.convert_to_unicode(self.language):
        continue
      text_a = tokenization.convert_to_unicode(line[6])
      text_b = tokenization.convert_to_unicode(line[7])
      label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]


class MnliProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
        "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "test_matched.tsv")),
      "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[8])
      text_b = tokenization.convert_to_unicode(line[9])
      if set_type == "test":
        label = "contradiction"
      else:
        label = tokenization.convert_to_unicode(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # Only the test set has a header
      if set_type == "test" and i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        label = "0"
      else:
        text_a = tokenization.convert_to_unicode(line[3])
        label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


# ======================================================
# ========== Data processor added by xxxx =============
class SnliProcessor(DataProcessor):
  """Processor for the SNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  @property
  def no_label_for_test(self):  # have label
    return False

  def get_labels(self):
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[7])
      text_b = tokenization.convert_to_unicode(line[8])
      if set_type == "test":
        label = tokenization.convert_to_unicode(line[-1])  # have label
      else:
        label = tokenization.convert_to_unicode(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class QqpProcessor(DataProcessor):
  """The data from https://github.com/zhiguowang/BiMPM"""
  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  @property
  def no_label_for_test(self):
    return False

  def get_labels(self):
    return ["0", "1"]

  def  _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[-1]).strip())
      text_a = tokenization.convert_to_unicode(line[1]).strip()
      text_b = tokenization.convert_to_unicode(line[2]).strip()
      if set_type == "test":
        label = tokenization.convert_to_unicode(line[0]).strip()  # have label
      else:
        label = tokenization.convert_to_unicode(line[0]).strip()
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class Sst2Processor(DataProcessor):
  def get_train_examples(self, data_dir):
    return self._create_examples(data_dir, 'train')

  def get_dev_examples(self, data_dir):
    return self._create_examples(data_dir, 'dev')

  def get_test_examples(self, data_dir):
    return self._create_examples(data_dir, 'test')

  @property
  def no_label_for_test(self):
    return False

  @property
  def is_paired_data(self):
    return False

  def get_labels(self):
    return ["0", "1"]

  def _create_examples(self, data_dir, set_type):

    only_sent = not (set_type == 'train')

    raw_data_list = _read_sst_tree_list(data_dir, set_type)
    data_list, _ = _gene_sst_sub_trees_and_shift_reduce_info(raw_data_list)
    examples = self._get_examples_from_data_list(data_list, only_sent, set_type)
    return examples

  def _sentiment2label(self, continous_sentiment_label):
    sentiment_label = None
    if continous_sentiment_label <= 0.4:
      sentiment_label = "0"
    elif continous_sentiment_label > 0.6:
      sentiment_label = "1"
    return sentiment_label

  def _get_examples_from_data_list(self, data_list, only_sentence, set_type):
    idx_ex = 0
    examples = []
    for sub_trees in data_list:
      for sub_tree in sub_trees:
        if only_sentence and not sub_tree['is_sent']:
          continue

        root_node = sub_tree['root_node']
        sentiment_label = self._sentiment2label(root_node['sentiment_label'])

        if sentiment_label is None:
          continue

        token_list = []
        for node in sub_tree['tree_nodes']:
          if node['is_leaf']:
            token_list.append(node['token'])

        assert len(token_list) > 0
        text_a = " ".join(token_list)
        # print(text_a)
        guid = "%s-%d" % (set_type, idx_ex)
        idx_ex += 1
        examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=sentiment_label))
    return examples


class Sst5Processor(Sst2Processor):
  def get_labels(self):
    return ["0", "1", "2", "3", "4"]

  def _sentiment2label(self, continous_sentiment_label):
    if continous_sentiment_label <= 0.2:
      sentiment_label = "0"
    elif continous_sentiment_label <= 0.4:
      sentiment_label = "1"
    elif continous_sentiment_label <= 0.6:
      sentiment_label = "2"
    elif continous_sentiment_label <= 0.8:
      sentiment_label = "3"
    elif continous_sentiment_label <= 1.:
      sentiment_label = "4"
    else:
      raise AttributeError(continous_sentiment_label)

    return sentiment_label


class TrecProcessor(Sst2Processor):
  def get_train_examples(self, data_dir):
    return self._create_examples(data_dir, 'train')

  def get_dev_examples(self, data_dir):
    return self._create_examples(data_dir, 'dev')

  def get_test_examples(self, data_dir):
    return []

  @property
  def no_label_for_test(self):
    return True

  @property
  def is_paired_data(self):
    return False

  def get_labels(self):
    return ["NUM", "LOC", "DESC", "ENTY", "ABBR", "HUM"]

  def _line2label(self, line):
    label = line.strip().split(' ')[0].split(':')[0]
    assert label in self.get_labels()
    return label

  def _create_examples(self, data_dir, set_type):
    if set_type == 'train':
      data_file_path = os.path.join(data_dir, "train_5500.label.txt")
    elif set_type == 'dev':
      data_file_path = os.path.join(data_dir, "TREC_10.label.txt")
    else:
      raise AttributeError

    examples = []
    with open(data_file_path, 'r', encoding='latin-1') as file:
      for idx_ex, line in enumerate(file):
        label = self._line2label(line)
        tokens = line.strip().split(' ')[1:]
        text_a = " ".join(tokens)
        guid = "%s-%d" % (set_type, idx_ex)

        examples.append(
          InputExample(
            guid=guid, text_a=text_a, label=label
          )
        )
    return examples


class Trec50Processor(TrecProcessor):
  def get_labels(self):
    return [
      'LOC:other', 'NUM:date', 'NUM:count', 'NUM:period', 'NUM:ord', 'NUM:other', 'ENTY:currency', 'LOC:state',
      'NUM:volsize', 'ENTY:plant', 'LOC:country', 'HUM:ind', 'ABBR:exp', 'ENTY:food', 'NUM:money', 'NUM:dist',
      'DESC:desc', 'HUM:desc', 'LOC:city', 'ENTY:termeq', 'LOC:mount', 'ENTY:word', 'ENTY:body', 'ENTY:dismed',
      'NUM:code', 'NUM:weight', 'NUM:temp', 'ENTY:product', 'HUM:title', 'DESC:def', 'DESC:manner', 'ENTY:animal',
      'ENTY:sport', 'ENTY:techmeth', 'NUM:speed', 'ENTY:veh', 'ENTY:religion', 'ENTY:instru', 'ENTY:other',
      'HUM:gr', 'DESC:reason', 'NUM:perc', 'ENTY:substance', 'ENTY:lang', 'ENTY:color', 'ENTY:cremat',
      'ENTY:event', 'ABBR:abb', 'ENTY:symbol', 'ENTY:letter'
    ]

  def _line2label(self, line):
    label = line.strip().split(' ')[0]
    assert label in self.get_labels()
    return label

# ======================================================

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id)
  return feature


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


# =========================== Other for Dataset ===============================

def _read_sst_tree_list(data_dir, set_type):
  assert set_type in ['train', 'dev', 'test']
  sentences_file = os.path.join(data_dir, 'datasetSentences.txt')
  dataset_split_file = os.path.join(data_dir, 'datasetSplit.txt')
  dictionary_file = os.path.join(data_dir, 'dictionary.txt')
  sentiment_labels_file = os.path.join(data_dir, 'sentiment_labels.txt')
  SOStr_file = os.path.join(data_dir, 'SOStr.txt')
  STree_file = os.path.join(data_dir, 'STree.txt')

  # dictionary: phrase2idx
  dictionary = {}
  with open(dictionary_file, encoding='utf-8') as file:
    for line in file:
      line = line.strip().split('|')
      assert len(line) == 2
      dictionary[line[0]] = int(line[1])

  # sentiment_labels: idx2label-str
  sentiment_labels = {}
  with open(sentiment_labels_file, encoding='utf-8') as file:
    file.readline()  # for table head
    for line in file:
      line = line.strip().split('|')
      sent_float_value = float(line[1])
      sentiment_labels[int(line[0])] = sent_float_value

  # STree.txt and SOStr.txt
  trees = []
  with open(STree_file, encoding='utf-8') as file_STree, \
          open(SOStr_file, encoding='utf-8') as file_SOStr:
    for STree, SOStr in zip(file_STree, file_SOStr):
      sent_tree = []
      STree = list(map(int, STree.strip().split('|')))
      SOStr = SOStr.strip().split('|')

      for idx_t, parent_idx in enumerate(STree):
        try:
          token = SOStr[idx_t]
          is_leaf = True
          leaf_node_index_seq = [idx_t + 1]
        except IndexError:
          token = ''
          is_leaf = False
          leaf_node_index_seq = []

        new_node = {'node_index': idx_t + 1, 'parent_index': parent_idx,
                    'token': token, 'is_leaf': is_leaf,
                    'leaf_node_index_seq': leaf_node_index_seq, }
        sent_tree.append(new_node)

      # update leaf_node_index_seq
      idx_to_node_dict = dict((tree_node['node_index'], tree_node)
                              for tree_node in sent_tree)
      for tree_node in sent_tree:
        if not tree_node['is_leaf']: break
        pre_node = tree_node
        while pre_node['parent_index'] > 0:
          cur_node = idx_to_node_dict[pre_node['parent_index']]
          cur_node['leaf_node_index_seq'] += pre_node['leaf_node_index_seq']
          cur_node['leaf_node_index_seq'] = list(
            sorted(list(set(cur_node['leaf_node_index_seq']))))
          pre_node = cur_node

      # update sentiment and add token_seq
      for tree_node in sent_tree:
        tokens = [sent_tree[node_idx - 1]['token'] for node_idx in tree_node['leaf_node_index_seq']]
        phrase = ' '.join(tokens)
        tree_node['sentiment_label'] = sentiment_labels[dictionary[phrase]]
        tree_node['token_seq'] = tokens

      trees.append(sent_tree)

  # dataset_split (head)  # list
  dataset_split = []
  with open(dataset_split_file, encoding='utf-8') as file:
    file.readline()  # for table head
    for line in file:
      dataset_split.append(int(line.strip().split(',')[1]))

  if set_type == 'train':
    target = 1
  elif set_type == 'test':
    target = 2
  else:
    target = 3

  data_list = []
  for _type, tree in zip(dataset_split, trees):
    if _type == target:
      data_list.append(tree)
  return data_list

def _gene_sst_sub_trees_and_shift_reduce_info(data_list):
  counter = 0
  new_data_list = []
  for tree in data_list:
    sub_trees = []
    idx_to_node_dict = dict((tree_node['node_index'], tree_node)
                            for tree_node in tree)
    for tree_node in tree:
      # get all node for a sub tree
      if tree_node['is_leaf']:
        new_sub_tree = [tree_node]
      else:
        new_sub_tree = []
        new_sub_tree_leaves = [idx_to_node_dict[node_index]
                               for node_index in tree_node['leaf_node_index_seq']]
        new_sub_tree += new_sub_tree_leaves
        for leaf_node in new_sub_tree_leaves:
          pre_node = leaf_node
          while pre_node['parent_index'] > 0 and pre_node != tree_node:  # fixme
            cur_node = idx_to_node_dict[pre_node['parent_index']]
            if cur_node not in new_sub_tree:
              new_sub_tree.append(cur_node)
            pre_node = cur_node
      # get shift reduce info
      child_node_indices = [new_tree_node['node_index'] for new_tree_node in new_sub_tree]
      parent_node_indices = [new_tree_node['parent_index']
                             if new_tree_node['parent_index'] in child_node_indices else 0
                             for new_tree_node in new_sub_tree]
      sr_result = shift_reduce_constituency_forest(list(zip(child_node_indices, parent_node_indices)))
      operation_list, node_list_in_stack, reduce_mat = zip(*sr_result)
      shift_reduce_info = {'op_list': operation_list,
                           'reduce_mat': reduce_mat,
                           'node_list_in_stack': node_list_in_stack}
      sub_tree = {'tree_nodes': new_sub_tree, 'shift_reduce_info': shift_reduce_info,
                  'root_node': tree_node, 'is_sent': True if tree_node['parent_index'] == 0 else False
                  }
      sub_trees.append(sub_tree)
      counter += 1
    new_data_list.append(sub_trees)
  return new_data_list, counter






