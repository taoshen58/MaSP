import re
import spacy
import ftfy


class GloveTextEncoder(object):
    def __init__(self, text_iter, glove_path, is_lower=True, default_corpus='6B', default_dim=300):
        self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
        self.glove_path = glove_path
        self.is_lower = is_lower

        # get token
        init_vocab = set()
        for text in text_iter:
            text_tokens = self._tokenize_sentence(text)
            init_vocab.update(set())
            all_token_list.extend(text_tokens)






    def _tokenize_sentence(self, text):
        text = self._standardize_text(ftfy.fix_text(text))
        text_tokens = []
        for token in self.nlp(text):
            text_token = token.text
            if self.is_lower:
                text_token = text_token.lower()
            text_tokens.append(text_token)
        return text_tokens


    def _standardize_text(self, text):
        text = text.replace('—', '-')
        text = text.replace('–', '-')
        text = text.replace('―', '-')
        text = text.replace('…', '...')
        text = text.replace('´', "'")
        text = re.sub('''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
        text = re.sub('\s*\n\s*', ' \n ', text)
        text = re.sub('[^\S\n]+', ' ', text)
        return text.strip()














