import re
import ftfy
import json
import spacy
from tqdm import tqdm


def get_pairs(word):  # read
    """
    Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def text_standardize(text):  # read
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub('''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub('\s*\n\s*', ' \n ', text)
    text = re.sub('[^\S\n]+', ' ', text)
    return text.strip()


class BpeTextEncoder(object):  # reading
    """
    mostly a wrapper for a public python bpe tokenizer
    """
    UNK_TOKEN = '<unk>'
    PAD_TOKEN = '@@@<pad>@@@'

    def __init__(self, encoder_path, bpe_path, add_padding=False, special_tokens=tuple()):  # read
        self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
        # add padding token or not
        self.encoder = json.load(open(encoder_path))  # this is subword2idx

        if add_padding:
            for k, v in self.encoder.items():
                self.encoder[k] += 1
            self.encoder[BpeTextEncoder.PAD_TOKEN] = 0

        # add special tokens
        for spe_token in special_tokens:
            self.encoder[spe_token] = len(self.encoder)

        self.decoder = {v:k for k,v in self.encoder.items()}  # this is idx2subword
        merges = open(bpe_path, encoding='utf-8').read().split('\n')[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    def bpe(self, token):  # read: bpe split
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)  # a list of pair

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        if word == '\n  </w>':
            word = '\n</w>'
        self.cache[token] = word
        return word

    def encode(self, texts, verbose=True):  # read: subword + tokenize
        texts_tokens = []
        if verbose:
            for text in tqdm(texts, ncols=80, leave=False):
                text = self.nlp(text_standardize(ftfy.fix_text(text)))
                text_tokens = []
                for token in text:
                    token_bpe = self.bpe(token.text.lower()).split(' ')
                    text_tokens.extend([self.encoder.get(t, 0) for t in token_bpe])
                texts_tokens.append(text_tokens)
        else:
            for text in texts:
                text = self.nlp(text_standardize(ftfy.fix_text(text)))
                text_tokens = []
                for token in text:
                    text_tokens.extend([self.encoder.get(t, 0) for t in self.bpe(token.text.lower()).split(' ')])
                texts_tokens.append(text_tokens)
        return texts_tokens

    def encode_sent(self, sent, max_seq_len):
        doc = self.nlp(text_standardize(ftfy.fix_text(sent)))
        text_tokens = []
        for token in doc:
            text_tokens.extend([t for t in self.bpe(token.text.lower()).split(' ')])
        if max_seq_len is not None:
            text_tokens = text_tokens[:max_seq_len]
        return text_tokens

    def get_idx_from_token(self, token):
        try:
            return self.encoder[token]
        except KeyError:
            return self.encoder[BpeTextEncoder.UNK_TOKEN]

    @property
    def n_vocab(self):
        return len(self.encoder)



