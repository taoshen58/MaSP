import spacy

spacy_nlp = spacy.load('en', disable=['parser', "ner", "tagger"])


def spacy_tokenize(text_input, lower=True):
    spacy_doc = spacy_nlp(text_input)
    return " ".join([token.lower_ if lower else token.text for token in spacy_doc])

if __name__ == '__main__':
    print(spacy_tokenize("How are you?"))

