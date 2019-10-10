#!/usr/bin/env python

import sys

from scipy.stats import spearmanr


DEFAULT_SEP = ';'


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-s', '--separator', default=DEFAULT_SEP,
                    help='character separating pairs (default "{}"'.format(
                        DEFAULT_SEP))
    ap.add_argument('gold')
    ap.add_argument('pred')
    return ap


def load_data(fn, options):
    data, seen = [], set()
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            fields = l.split(options.separator)
            if len(fields) < 3:
                print('format error on line {} in {}: {}'.format(ln, fn, l),
                      file=sys.stderr)
                raise ValueError
            w1, w2, sim = fields[:3]
            if (w1, w2) in seen:
                print('dup pair on line {} in {}: {}'.format(ln, fn, l),
                      file=sys.stderr)
                continue
            try:
                sim = float(sim)
            except ValueError:
                print('invalid value on line {} in {}: {}'.format(ln, fn, l),
                      file=sys.stderr)
                continue
            data.append((w1, w2, sim))
            seen.add((w1, w2))
    return data


def filter_data(data, other, name):
    pairs = set([(w1,w2) for w1, w2, sim in other])
    filtered = []
    for w1, w2, sim in data:
        if (w1, w2) not in pairs:
            print('{} only in {}, skipping'.format((w1,w2), name))
        else:
            filtered.append((w1, w2, sim))
    return filtered


def main(argv):
    args = argparser().parse_args(argv[1:])
    try:
        gold = load_data(args.gold, args)
        pred = load_data(args.pred, args)
    except ValueError:
        return 1
    gold = filter_data(gold, pred, 'gold')
    pred = filter_data(pred, gold, 'pred')
    print('evaluating {} gold vs {} pred'.format(len(gold), len(pred)))
    sorted_gold = sorted(gold, key=lambda i: i[2], reverse=True)
    pred_by_pair = { (w1, w2): sim for w1, w2, sim in pred }
    sorted_pred = []
    for w1, w2, sim in sorted_gold:
        sorted_pred.append((w1, w2, pred_by_pair[(w1, w2)]))
    a = [i[2] for i in sorted_gold]
    b = [i[2] for i in sorted_pred]
    correlation, pvalue = spearmanr(a, b)
    print(correlation)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))


