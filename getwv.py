#!/usr/bin/env python

import sys
import json
import numpy as np

from logging import warning


IGNORE = set(['[CLS]', '[SEP]'])


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('file', nargs='+', metavar='JSONL',
                    help='BERT extract_features.py output')
    return ap


def is_continuation(token):
    return token.startswith('##')


def save_vector(pieces, values, vectors):
    if not pieces:
        raise ValueError('no pieces')
    word = pieces[0]
    for p in pieces[1:]:
        word += p[2:]
    v = np.mean(values, axis=0)
    if word not in vectors:
        vectors[word] = (v, 1)
    else:
        vectors[word] = (vectors[word][0]+v, vectors[word][1]+1)


def load_vectors(fn, vectors, options):
    total = 0
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            data = json.loads(l)
            curr_pieces, curr_values = [], []
            for d in data['features']:             
                token = d['token']
                if token in IGNORE:
                    continue
                layers = d['layers']
                if len(layers) != 1:
                    warning('expected one layer, got {}, ignoring all '
                            'but first'.format(len(l)))
                values = np.array(layers[0]['values'])
                if is_continuation(token):
                    if not curr_pieces:
                        raise ValueError('line-initial "{}"'.format(token))
                    curr_pieces.append(token)
                    curr_values.append(values)
                else:
                    # not continuation, i.e. new token
                    if curr_pieces:
                        save_vector(curr_pieces, curr_values, vectors)
                        #v = np.mean(curr_values, axis=0)
                        #vectors_by_token[curr_token].append(v)
                        total += 1
                    curr_pieces = [token]
                    curr_values = [values]
            # process last
            if curr_pieces is not None:
                save_vector(curr_pieces, curr_values, vectors)
                #v = np.mean(curr_values, axis=0)
                #vectors_by_token[curr_token].append(v)
                total += 1
    print('loaded {} vectors for {} tokens from {}'.format(
        total, len(vectors), fn), file=sys.stderr)
    return vectors


def main(argv):
    args = argparser().parse_args(argv[1:])

    # maintain a running sum and count of vectors per word
    vectors = {}
    for fn in args.file:
        vectors_by_token = load_vectors(fn, vectors, args)
    # divide by count for average vector
    vectors = { w: v/c for w, (v, c) in vectors.items() }

    if not vectors:
        warning('no vectors loaded, exiting')
        return 1

    tokens = len(vectors)
    dim = len(next(iter(vectors.values())))
    print(tokens, dim) # word2vec header line
    for t, v in vectors.items():
        #v = np.mean(vectors, axis=0)
        v /= np.linalg.norm(v)
        print(t, ' '.join('{:.4f}'.format(i) for i in v))

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
