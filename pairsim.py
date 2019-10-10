#!/usr/bin/env python3

import sys
import numpy as np

from gensim.models import KeyedVectors


DEFAULT_SEP = ';'


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-s', '--separator', default=DEFAULT_SEP,
                    help='character separating pairs (default "{}"'.format(
                        DEFAULT_SEP))
    ap.add_argument('wordvecs')
    ap.add_argument('pairs')
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])
    wv = KeyedVectors.load_word2vec_format(args.wordvecs, binary=False)
    similarities = []
    with open(args.pairs) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            try:
                w1, w2 = l.split(args.separator)
            except ValueError:
                print('expected two words separated by "{}" on line {} in {}, '
                      'got {}'.format(args.separator, ln, args.pairs, l))
                return 1
            if w1 not in wv:
                print('{} not found, skipping {}'.format(w1, (w1, w2)),
                      file=sys.stderr)
            elif w2 not in wv:
                print('{} not found, skipping {}'.format(w2, (w1, w2)),
                      file=sys.stderr)
            else:
                sim = np.dot(wv[w1], wv[w2])
                similarities.append((sim, w1, w2))
    for sim, w1, w2 in sorted(similarities, reverse=True):
        print('{}'.format(args.separator.join((w1, w2, str(sim)))))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
