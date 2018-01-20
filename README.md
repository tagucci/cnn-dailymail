This code produces the non-anonymized version of the CNN / Daily Mail summarization dataset, as used in the ACL 2017 paper *[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/pdf/1704.04368.pdf)*.

***This is a modified scripts to obtain raw texts instead of tensorflow binaries.***

# Instructions

## 1. Download data
Download and unzip the `stories` directories from [here](http://cs.nyu.edu/~kcho/DMQA/) for both CNN and Daily Mail.

## 2. Download Stanford CoreNLP
We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Then add the following command to your bash_profile:
```
export CLASSPATH=/path/to/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar
```
replacing `/path/to/` with the path to where you saved the `stanford-corenlp-full-2016-10-31` directory. You can check if it's working by running
```
echo "Please tokenize this text." | java edu.stanford.nlp.process.PTBTokenizer
```
You should see something like:
```
Please
tokenize
this
text
.
PTBTokenizer tokenized 5 tokens at 68.97 tokens per second.
```

## 3. Process into .txt and vocab files
Run
```
python make_datafiles.py /path/to/cnn/stories /path/to/dailymail/stories
```
replacing `/path/to/cnn/stories` with the path to where you saved the `cnn/stories` directory that you downloaded; similarly for `dailymail/stories`.

This script will do several things:
- First, `cnn_stories_tokenized` and `dm_stories_tokenized` will be temporary created and filled with tokenized versions of `cnn/stories` and `dailymail/stories`. If you set `rm_tokenized_dir = True`, these tokenized directories will be removed after processing texts. This may take some time. ***Note***: *you may see several `Untokenizable:` warnings from Stanford Tokenizer. These seem to be related to Unicode characters in the data; so far it seems OK to ignore them.*

- For each of the url lists `all_train.txt`, `all_val.txt` and `all_test.txt`, the corresponding tokenized stories are read from file, lowercased and written to text files.

- All text files  saved in newly-created `finished_files` directory. There will be `{train/val/test}_{article/abstract}.txt` in that directory. This text format is written each stories article/abstract line by line. This allows you to quickly try open source packages like *[OpenNMT](http://opennmt.net/)* to train models.

- The original data size is `train: 287,226`, `val: 13,368`, `test: 11,490` as described in the paper. *However, `train: 287,226` because train dataset contains 114 empty articles.*

- In addition, `train/val/test` directories will be created in `finished_files`. In each directory, stories of article and abstract are placed in `article/abstract` directories. Both of them are saved line by line. Considering extractive summarization methods, this may be convenient.

- Additionally, a `vocab` file is created from the training data. This is also placed in `finished_files`.

### Extra (Lead-3 baseline result)

I evaluate lead-3 baseline as *[See's paper](https://arxiv.org/pdf/1704.04368.pdf)* showed.
Instead of using *[pyrouge](https://pypi.python.org/pypi/pyrouge/0.1.3)* as the author used, I use *[pythonrouge](https://github.com/tagucci/pythonrouge)* to evaluate ROUGE.
***While I got same ROUGE scores of pointer-generator / pointer-generator+coverage models by using test-output downloaded from author's *[pointer-generator repository]( https://github.com/abisee/pointer-generator)*, lead-3 baseline result is slightly different.***

If you want to evaluate lead-3 baseline, `eval_rouge = True` in line 19 of `make_datafiles.py`. You also need to install pythonrouge package.
```
# install pythonrouge
pip install git+https://github.com/tagucci/pythonrouge.git
```

#### ROUGE Scores


|                                          | ROUGE-1   | ROUGE-2   | ROUGE-L   |
|------------------------------------------|-----------|-----------|-----------|
| lead-3 baseline (Nallapati et al., 2017) | 39.2      | 15.7      | 35.5      |
| lead-3 baseline (See et al., 2017)       | **40.34** | **17.70** | **36.57** |
| lead-3 baseline (this repository)        | 40.24     | **17.70** | 36.45     |
