#!/usr/bin/env python

import sys
import json
import numpy as np

from collections import defaultdict
from logging import warning


IGNORE = set(['[CLS]', '[SEP]'])


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('file', nargs='+', metavar='JSONL',
                    help='BERT extract_features.py output')
    return ap


def load_vectors(fn, options):
    vectors_by_token = defaultdict(list)
    total = 0
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            data = json.loads(l)
            for d in data['features']:             
                token = d['token']
                if token in IGNORE:
                    continue
                layers = d['layers']
                if len(layers) != 1:
                    warning('expected one layer, got {}, ignoring all but first'.format(len(l)))
                v = np.array(layers[0]['values'])
                vectors_by_token[token].append(v)
                total += 1
    print('loaded {} vectors for {} tokens from {}'.format(
        total, len(vectors_by_token), fn), file=sys.stderr)
    return vectors_by_token


def main(argv):
    args = argparser().parse_args(argv[1:])
    combined_by_token = defaultdict(list)
    for fn in args.file:
        vectors_by_token = load_vectors(fn, args)
        for t, v in vectors_by_token.items():
            combined_by_token[t].extend(v)

    if not combined_by_token:
        warning('no vectors loaded, exiting')
        return 1

    tokens = len(combined_by_token)
    dim = len(next(iter(combined_by_token.values()))[0])
    print(tokens, dim) # word2vec header line
    for t, vectors in combined_by_token.items():
        v = np.mean(vectors, axis=0)
        v /= np.linalg.norm(v)
        print(t, ' '.join('{:.4f}'.format(i) for i in v))

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
