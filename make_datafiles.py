import collections
from datetime import datetime
import hashlib
import os
from pprint import pprint
from shutil import rmtree
import subprocess
import sys


dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"',dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
# Evaluate ROUGE score of test-set
eval_rouge = False

all_train_urls = "url_lists/all_train.txt"
all_val_urls = "url_lists/all_val.txt"
all_test_urls = "url_lists/all_test.txt"

cnn_tokenized_stories_dir = "cnn_stories_tokenized"
dm_tokenized_stories_dir = "dm_stories_tokenized"
finished_files_dir = "finished_files"
# Remove cnn/dailymail tokenized dirs
rm_tokenized_dir = True

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506

VOCAB_SIZE = 200000


def printLog(message):
    print("[{}] {}".format(datetime.now().strftime("%D %H:%M:%S"), message))


def tokenize_stories(stories_dir, tokenized_stories_dir):
    """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
    printLog("Preparing to tokenize {} to {}..." .format(stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    printLog("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
        for s in stories:
            f.write("{} \t {}\n".format(os.path.join(stories_dir, s),
                                        os.path.join(tokenized_stories_dir, s)))
        command = ['java',
                   'edu.stanford.nlp.process.PTBTokenizer',
                   '-ioFileList',
                   '-preserveLines',
                   'mapping.txt']
    printLog("Tokenizing {} files in {} and saving in {}..." .format(len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    printLog("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping.txt")
    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception("The tokenized stories directory {} contains {} files,\
                        but it should contain the same number as {} (which has\
                        {} files). Was there an error during tokenization?".format(tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    printLog("Successfully finished tokenizing {} to {}.\n".format(stories_dir, tokenized_stories_dir))


def read_text_file(text_file):
    with open(text_file) as f:
        lines = f.read().strip().split('\n')
    return lines


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url.encode('utf-8')) for url in url_list]


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line:
        return line
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + " ."


def get_art_abs(story_file):
    lines = read_text_file(story_file)

    # Lowercase everything
    lines = [line.lower() for line in lines]

    # Put periods on the ends of lines that are missing them
    # This is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article & abstract
    article = '\n'.join(article_lines)
    abstract = '\n'.join(highlights)

    # Article_lines is used when evaluating LEAD
    return article, abstract, article_lines


def write_to_txt(url_file, out_file, makevocab=False):
    """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
    printLog("Making txt file for URLs listed in {}...".format(url_file))
    url_list = read_text_file(url_file)
    url_hashes = get_url_hashes(url_list)
    story_fnames = [s+".story" for s in url_hashes]
    num_stories = len(story_fnames)
    empty_files = 0
    if makevocab:
        vocab_counter = collections.Counter()

    # train/val/test
    data_split = out_file.split('/')[-1]
    # Data list to write article & abstract to .txt file
    contents = []
    # To evaluate ROUGE of LEAD
    if eval_rouge and data_split == 'test':
        lead_data = []

    for idx, s in enumerate(story_fnames):
        # Look in the tokenized story dirs to find the .story file corresponding to this url
        if os.path.isfile(os.path.join(cnn_tokenized_stories_dir, s)):
            story_file = os.path.join(cnn_tokenized_stories_dir, s)
        elif os.path.isfile(os.path.join(dm_tokenized_stories_dir, s)):
            story_file = os.path.join(dm_tokenized_stories_dir, s)
        else:
            printLog("Error: Couldn't find tokenized story file {} in either tokenized story directories {} and {}. Was there an error during tokenization?".format(s, cnn_tokenized_stories_dir, dm_tokenized_stories_dir))
            # Check again if tokenized stories directories contain correct number of files
            printLog("Checking that the tokenized stories directories {} and {} contain correct number of files...".format(cnn_tokenized_stories_dir, dm_tokenized_stories_dir))
            check_num_stories(cnn_tokenized_stories_dir, num_expected_cnn_stories)
            check_num_stories(dm_tokenized_stories_dir, num_expected_dm_stories)
            raise Exception("Tokenized stories directories {} and {} contain correct number of files but story file {} found in neither." % (cnn_tokenized_stories_dir, dm_tokenized_stories_dir, s))
        # Get the strings to write to .txt file
        article, abstract, lead_sents = get_art_abs(story_file)
        contents.append([s, article, abstract])
        # When data_split is test-set, evaluating LEAD
        if eval_rouge and data_split == 'test':
            # Lead-3sent
            lead_data.append([lead_sents[:3], abstract.split('\n')])

        # Write the vocab to file, if applicable
        if makevocab:
            art_tokens = article.split(' ')
            abs_tokens = abstract.split(' ')
            abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
            tokens = art_tokens + abs_tokens
            tokens = [t.strip() for t in tokens]  # strip
            tokens = [t for t in tokens if t != ""]  # remove empty
            vocab_counter.update(tokens)

    # write article & abstract in each stories
    if not os.path.isdir(out_file):
        os.mkdir(out_file)
    article_dir = os.path.join(out_file, 'article')
    if not os.path.isdir(article_dir):
        os.mkdir(article_dir)
    abstract_dir = os.path.join(out_file, 'abstract')
    if not os.path.isdir(abstract_dir):
        os.mkdir(abstract_dir)

    # Write article & abstract line by line
    atfname = '{}_article.txt'.format(out_file)
    abfname = '{}_abstract.txt'.format(out_file)
    arti = open(atfname, 'w')
    abst = open(abfname, 'w')

    for idx, content in enumerate(contents):
        # story_file, article, abstract
        s, at, ab = content
        if len(at) != 0:
            # Save article & abstract of each story article & abstract dir
            with open('{}/{}'.format(article_dir, s), 'w') as f:
                for a in at.split('\n'):
                    f.write('{}\n'.format(a))
            with open('{}/{}'.format(abstract_dir, s), 'w') as f:
                for a in ab.split('\n'):
                    f.write('{}\n'.format(a))

            # Save article & abstract line by line in a file
            concat_arti = ' '.join(at.split())
            arti.write('{}\n'.format(concat_arti))
            concat_abst = ' '.join(ab.split())
            abst.write('{}\n'.format(concat_abst))
        else:
            empty_files += 1

    # Write vocab to file
    if makevocab:
        printLog("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write('{} {}\n'.format(word, count))

    printLog('total files in {} : {}({} files are empty)\n'.format(data_split, len(contents), empty_files))
    printLog("Finished writing files:\n{}\n{}\n".format(atfname, abfname))
    if eval_rouge and data_split == 'test':
        printLog('ROUGE evaluation start...')
        from pythonrouge.pythonrouge import Pythonrouge
        leads = []
        reference = []
        for data in lead_data:
            lead, summary = data
            leads.append(lead)
            reference.append([summary])
        rouge = Pythonrouge(summary_file_exist=False,
                            # if you want to check setting file of ROUGE, set delete_xml=False
                            delete_xml=True,
                            summary=leads, reference=reference,
                            n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                            f_measure_only=True,
                            stemming=True, stopwords=False,
                            word_level=True, length_limit=False)
        score = rouge.calc_score()
        printLog(data_split)
        pprint(score)




def check_num_stories(stories_dir, num_expected):
    num_stories = len(os.listdir(stories_dir))
    if num_stories != num_expected:
        raise Exception("stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        printLog("USAGE: python make_datafiles.py <cnn_stories_dir> <dailymail_stories_dir>")
        sys.exit()
    cnn_stories_dir = sys.argv[1]
    dm_stories_dir = sys.argv[2]
    # Check the stories directories contain the correct number of .story files
    check_num_stories(cnn_stories_dir, num_expected_cnn_stories)
    check_num_stories(dm_stories_dir, num_expected_dm_stories)
    # Create some new directories
    if not os.path.exists(cnn_tokenized_stories_dir):
        os.makedirs(cnn_tokenized_stories_dir)
    if not os.path.exists(dm_tokenized_stories_dir):
        os.makedirs(dm_tokenized_stories_dir)
    if not os.path.exists(finished_files_dir):
        os.makedirs(finished_files_dir)

    # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
    tokenize_stories(cnn_stories_dir, cnn_tokenized_stories_dir)
    tokenize_stories(dm_stories_dir, dm_tokenized_stories_dir)

    # Read the tokenized stories, do a little postprocessing then write to txt files
    # 11,490 files
    write_to_txt(all_test_urls, os.path.join(finished_files_dir, "test"))
    # 13,368 files
    write_to_txt(all_val_urls, os.path.join(finished_files_dir, "val"))
    # 287,227 files, but 114 articles are empty
    # 287,113 files are saved
    write_to_txt(all_train_urls, os.path.join(finished_files_dir, "train"), makevocab=True)

    # Remove tokenized_stories_dir
    if rm_tokenized_dir:
        rmtree(cnn_tokenized_stories_dir)
        rmtree(dm_tokenized_stories_dir)
