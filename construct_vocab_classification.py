#!/usr/bin/env python

from __future__ import print_function

import argparse, json, io, ast

from text_classification import text_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-data', default='imdb.binary',
                        choices=['dbpedia', 'imdb.binary', 'imdb.fine',
                                 'TREC', 'stsa.binary', 'stsa.fine',
                                 'custrev', 'mpqa', 'rt-polarity', 'subj'],
                        help='Name of dataset.')
    parser.add_argument('--train-path')
    parser.add_argument('--test-path')
    parser.add_argument('--vocab-path')
    parser.add_argument('--tokens-to-add')
    args = parser.parse_args()
    construct(args)


def construct(args):
    vocab = None

    tokens_to_add = None
    if args.tokens_to_add:
        with io.open(args.tokens_to_add, 'r', encoding='utf8') as f:
            tokens_to_add = ast.literal_eval(f.read())

    # Load a dataset
    if args.dataset == 'dbpedia':
        train, test, vocab = text_datasets.get_dbpedia(
            vocab=vocab)
    elif args.dataset.startswith('imdb.'):
        train, test, vocab = text_datasets.get_imdb(
            fine_grained=args.dataset.endswith('.fine'),
            vocab=vocab)
    elif args.dataset in ['TREC', 'stsa.binary', 'stsa.fine',
                          'custrev', 'mpqa', 'rt-polarity', 'subj']:
        train, test, vocab = text_datasets.get_other_text_dataset(
            args.dataset, args.train_path, args.test_path, vocab=vocab, tokens_to_add=tokens_to_add)

    with io.open(args.vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)


if __name__ == '__main__':
    main()