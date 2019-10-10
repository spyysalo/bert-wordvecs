# BERT wordvecs

Tools for generating word vectors with BERT

## Quickstart

Clone BERT

```
git clone https://github.com/google-research/bert.git
export BERT_DIR=$(pwd)/bert
```

Download a BERT model (here FinBERT)

```
wget http://dl.turkunlp.org/finbert/bert-base-finnish-cased.zip
unzip bert-base-finnish-cased.zip 
export MODEL_DIR=$(pwd)/bert-base-finnish-cased
```

Create `input.txt` with some text

```
echo 'sanoja: auto, hissi, silta, letku, katu' > input.txt
```

Run BERT `extract_features.py` to extract contextual embeddings

```
python $BERT_DIR/extract_features.py \
  --do_lower_case=false \
  --input_file=input.txt \
  --output_file=output.jsonl \
  --vocab_file=$MODEL_DIR/vocab.txt \
  --bert_config_file=$MODEL_DIR/bert_config.json \
  --init_checkpoint=$MODEL_DIR/bert-base-finnish-cased \
  --layers=-1 \
  --max_seq_length=128 \
  --batch_size=8
```

Get average word vectors, convert to word2vec text format

```
python3 getwv.py output.jsonl > output.wv
```

Test with gensim

```
python3
>>> import numpy as np
>>> from gensim.models import KeyedVectors
>>> wv = KeyedVectors.load_word2vec_format('output.wv', binary=False)
>>> np.dot(wv['auto'], wv['katu'])
```

## Experiments

(Assuming the above is set up)

Clone the FinSemEvl repository

```
git clone https://github.com/venekoski/FinSemEvl.git
```

Grab word pairs and unique words from similarity dataset

```
tail -n +2 FinSemEvl/FinSemEvl/FinnSim/FinnSim_judgment_scores.csv \
    | cut -d ';' -f 1,2 > pairs.txt
tr ';' '\n' < pairs.txt | sort | uniq > input.txt
```

Run `extract_features.py`

```
python $BERT_DIR/extract_features.py \
  --do_lower_case=false \
  --input_file=input.txt \
  --output_file=output.jsonl \
  --vocab_file=$MODEL_DIR/vocab.txt \
  --bert_config_file=$MODEL_DIR/bert_config.json \
  --init_checkpoint=$MODEL_DIR/bert-base-finnish-cased \
  --layers=0 \
  --max_seq_length=128 \
  --batch_size=8
```

Convert

```
python3 getwv.py output.jsonl > output.wv
```

Get word pair similarities

```
python3 pairsim.py output.wv pairs.txt > ranked.txt
```

Compare

```
python3 correlation.py FinSemEvl/FinSemEvl/FinnSim/FinnSim_judgment_scores.csv ranked.txt | tail -n 1
```
