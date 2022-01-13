'''
NER using spaCy.
'''

from collections import Counter
from functools import reduce
from glob import glob
import logging
from pathlib import Path

import spacy


def cleanup_texts(text):
    '''Replace "etamp;" with "&" in the plain text.'''
    return text.replace('etamp;', '&')


def chunk_text(text, tokenizer, max_len):
    '''Split a text up into chunks of a given word count.
    Returns a list of strings, even if only a single chunk is created.
    '''
    doc = tokenizer(text)
    if max_len is not None:
        yield from (doc[i:i+max_len] for i in range(0, len(doc), max_len))
    else:
        yield doc[:]


def get_ents_count(doc, variant='LOC'):
    '''Return most common named entities of given variant.'''
    return Counter([
        ent.text
        for ent in doc.ents
        if ent.label_ == variant
    ])


def get_nlp_loader(iterations=None, doc_length=2_800_000):
    '''Memoize and periodically reload the language model to clear the vocab and string store memory.

    The reloading every N iterations drastically increases runtime and is therefore disabled by default.
    However, it can save some memory in case spaCy's own String Store gets to large.
    '''
    # The larger, but more accurate pre-trained french language model has to be installed with
    # `python -m spacy download fr_core_news_lg`
    nlp = spacy.load("fr_core_news_lg", exclude=['tagger', 'parser', 'lemmatizer', 'textcat'])
    nlp.max_length = doc_length
    tokenizer = nlp.tokenizer
    it = 1

    def loader():
        nonlocal it
        nonlocal nlp
        nonlocal tokenizer

        if iterations is not None and it % iterations == 0:
            logging.warning('reloading language model')
            nlp = spacy.load("fr_core_news_lg", exclude=['tagger', 'parser', 'lemmatizer', 'textcat'])
            nlp.max_length = doc_length
            tokenizer = nlp.tokenizer
        else:
            logging.info('reuse language model')

        it += 1
        return nlp, tokenizer
    
    return loader
        

def get_texts(pattern):
    '''Generator which yields all the file paths matching a given glob pattern.'''
    for path in glob(pattern):
        with open(path, encoding='utf8') as f:
            text = f.read()
        name = Path(path).name
        text = cleanup_texts(text)
        yield (name, text)


def sum_up_counters(counter_list):
    '''Given a list of `Counter`s, add all the counts up and
    return a comprehensive counter.
    '''
    return reduce(lambda a, b: a+b, counter_list)