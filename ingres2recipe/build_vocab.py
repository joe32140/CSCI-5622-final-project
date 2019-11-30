import os
import pickle
import json
import argparse
from collections import Counter
import numpy as np
import re
import pandas as pd
import Constants
from nltk.tokenize import TweetTokenizer

nlp = TweetTokenizer()
class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.word_count = []

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(text, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()

    data = json.load(open(text))
    print(len(data))
    for i, d in enumerate(data):
        if i % 500 == 0 :
            print(i)
        recipe = d[1]
        ingres = " ".join(d[2])
        text = recipe + ingres
        text = text.lower()
        tokens = nlp.tokenize(text)
        #tokens = [t for t in tokens]
        #print(tokens)
        counter.update(tokens)
    print(data[1])

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    # Creates a vocab wrapper and add some special .
    vocab = Vocabulary()
    word_count = {}
    for word, cnt in counter.items():
        word_count[word] = cnt
    # Adds the words to the vocabulary.
    
    vocab.add_word(Constants.PAD_WORD)
    vocab.add_word(Constants.UNK_WORD)
    vocab.add_word(Constants.BOS_WORD)
    vocab.add_word(Constants.EOS_WORD)
    for i, word in enumerate(words):
        vocab.add_word(word)
    for word, idx in vocab.word2idx.items():
        if word in word_count.keys():
            count = word_count[word]
            vocab.word_count.append(1/count)
        else:
            vocab.word_count.append(int(1))
    return vocab

def main(args):
    if not os.path.exists(args.vocab_dir):
        os.makedirs(args.vocab_dir)
        print("Make Data Directory")
    vocab = build_vocab(text=args.data_path,
                        threshold=args.threshold)
    #W = build_glove_voc(len(vocab), vocab, args.paragraph)
    vocab_path = os.path.join(args.vocab_dir, f'vocab.pkl')
    #weight_path = os.path.join(args.vocab_dir, 'W.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    #with open(weight_path, 'wb') as f:
    #    pickle.dump(W, f)

    print("Total vocabulary size: %d" %len(vocab))
    print(vocab.word2idx)
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_dir', type=str, default='./',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--data_path', type=str, default='train.json',
                         help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=3,
                        help='minimum word count threshold')

    args = parser.parse_args()
    main(args)
