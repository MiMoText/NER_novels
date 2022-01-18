#!/usr/bin/env python3

'''
NER_novels.py

Parse a directory of plain text files, find named entities
with spaCy, and store the results in a .csv table.
'''

import argparse
import logging

import pandas as pd

from src import ner
from src.utils import nullable_number

# Adjust the following config variables as needed.
# Every option can also be set directly on the CLI.

# The directory where the plain text files are stored, and where
# the eventual results should be stored.
SOURCES_PATH = '../../roman18/XML-TEI/files/'
RESULTS_PATH = '../raw_results/ner_loc_per.csv'

# On machines with limited memory, some of the larger
# documents might cause OOM. To prevent that, it is
# possible to split the texts into chunks of the given
# word count. Set to `None` to disable chunking.
CHUNK_SIZE = 120_000

# spaCy uses a maximum text length (in characters) for
# at document creation to prevent OOM. The default
# is 1_000_000. If `CHUNK_SIZE` is used, it will probably
# prevent the docs from ever reaching this size.
MAX_DOC_LENGTH = 2_800_000 

# Reload the spaCy language model every N iterations.
# Drastically increases runtime, but can save a bit of
# memory because it also clears spaCy's internal String
# Store. Set to `None` to disable model reloading.
# Since this should not normally be our bottleneck, the
# default setting is `None`.
RELOAD_MODEL = None

# The number of most frequent NEs to store.
MOST_FREQUENT_COUNT = 5



def main(args):
    results = {}
    nlp_loader = ner.get_nlp_loader(iterations=args.reload_model, doc_length=args.max_doc_length)
    sources = args.sources_path if args.sources_path.endswith('.txt') else f'{args.sources_path}*.txt'

    for name, text in ner.get_texts(sources):
        logging.warning(f'Working on text {name}')
        _, tokenizer = nlp_loader()

        # Keep track of the most frequent NEs of each chunk.
        loc_counters = []
        per_counters = []
        for chunk in ner.chunk_text(text, tokenizer, args.chunk_size):
            logging.debug('got chunk')
            logging.debug(len(chunk))
            nlp, _ = nlp_loader()
            logging.debug(nlp)
            doc = nlp(chunk.text)            
            loc_counters.append(ner.get_ents_count(doc))
            logging.debug('counted LOC')
            per_counters.append(ner.get_ents_count(doc, 'PER'))
            logging.debug('counted PER')
        
        logging.info('have all subcounts')
        most_freq_loc = ner.sum_up_counters(loc_counters)
        most_freq_per = ner.sum_up_counters(per_counters)
        results[name] = (
            most_freq_loc.most_common(args.most_frequent_count),
            most_freq_per.most_common(args.most_frequent_count))
        
    # Organize data as pandas DataFrame and write to csv file.
    df = pd.DataFrame.from_dict(results, orient='index', columns=['LOC', 'PER'])
    df.to_csv(args.results_path)
    logging.warning(f'Written results to {args.results_path}')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Use SpaCy to get the most frequently used NEs in french novels.')
    arg_parser.add_argument(
        '-s', '--sources-path',
        metavar='SOURCES_PATH',
        default=SOURCES_PATH,
        help='path to the directory where the plain text input files are, or to one specific txt file')
    arg_parser.add_argument(
        '-r' ,'--results-path',
        metavar='RESULTS_PATH',
        default=RESULTS_PATH,
        help='path of .csv file where the results will be written to')
    arg_parser.add_argument(
        '--chunk-size',
        type=nullable_number,
        metavar='WORDCOUNT',
        default=CHUNK_SIZE,
        help='Optionally split longer texts into chunks with this word count to reduce memory usage.')
    arg_parser.add_argument(
        '--max-doc-length',
        type=int,
        metavar='CHARCOUNT',
        default=MAX_DOC_LENGTH,
        help='Maximum length of documents (in characters) which SpaCy will accept')
    arg_parser.add_argument(
        '--reload-model',
        type=nullable_number,
        default=RELOAD_MODEL,
        metavar='I',
        help='Optionally reload spaCy language model every I iterations.'
        )
    arg_parser.add_argument(
        '-c', '--most-frequent-count',
        default=MOST_FREQUENT_COUNT,
        metavar='COUNT',
        type=int,
        help='the number of most frequent NEs to store'
        )
    arg_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='wether to use more verbose logging output')

    args = arg_parser.parse_args()

    loglevel = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(format='%(asctime)s -- %(message)s', datefmt='%Y-%m-%d %I:%M:%S', level=loglevel)

    logging.info('running with the following options')
    logging.info(args)
    main(args)