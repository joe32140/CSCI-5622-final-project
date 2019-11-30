import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from build_vocab import Vocabulary
import Constants
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import os,sys
from nltk.tokenize import TweetTokenizer
nlp = TweetTokenizer()

class Ingres2RecipeDataset(Dataset):
    def __init__(self, vocab, data, feature=None, period="24"):
        self.data = data
        self.vocab = vocab
        self.max_len = 400

    def __getitem__(self, index):
        # 0: recipe_id, 1:instruction/recipe, 2:list of ingres
        recipe = self.data[index][1]
        ingres = self.data[index][2]
        ingres = " ".join(ingres)

        # tokenize recipe
        r_tokens = []
        tokenized_r = nlp.tokenize(recipe)
        r_tokens.extend([self.vocab(token) for i, token in enumerate(tokenized_r) if i<self.max_len])

        # tokenize ingres
        i_tokens = []
        tokenized_i = nlp.tokenize(ingres)
        i_tokens.extend([self.vocab(token) for i, token in enumerate(tokenized_i) if i<self.max_len])


        return r_tokens, i_tokens

    def collate_fn(self, data):
        recipes, ingres = zip(*data)
        r_lengths = [len(x) for x in recipes]
        i_lengths = [len(x) for x in ingres]

        padded_recipes = [n + [Constants.PAD for _ in range(self.max_len - len(n))] for n in recipes]
        padded_ingres = [n + [Constants.PAD for _ in range(self.max_len - len(n))] for n in ingres]

        recipes = torch.LongTensor(padded_recipes).view(-1, self.max_len)
        ingres = torch.LongTensor(padded_ingres).view(-1, self.max_len)
        r_lengths = torch.LongTensor(r_lengths).view(-1,1)
        i_lengths = torch.LongTensor(i_lengths).view(-1,1)
        return recipes, ingres, r_lengths, i_lengths

    def __len__(self):
        return len(self.listfiles)

def get_loader(data, vocab, batch_size, shuffle, num_workers):
    ingres2recipe = Ingres2RecipeDataset(vocab, data)

    data_loader = torch.utils.data.DataLoader(dataset=ingres2recipe,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=Ingres2RecipeDataset.collate_fn)
    return data_loader

def get_loaders(args, is_test=False, is_feature=False):
    print(f"Note name : {args.name}")
    with open(f"vocab.pkl",'rb') as f:
        print("----- Loading Vocab -----")
        vocab = pickle.load(f)
        print(f"vocab size: {len(vocab)}")
    print('----- Loading Note -----')

    train = json.load(open('train.json'))
    valid = json.load(open('val.json'))
    test = json.load(open('test.json'))

    #train, valid = train_test_split(train, test_size=0.2, random_state=19)
    print("train size", len(train))
    print("val size", len(valid))
    print("test size", len(test))
    print()
    print('----- Building Loaders -----')
    train_loader = get_loader(train, vocab, args.batch_size, True, 10)
    valid_loader = get_loader(valid, vocab, args.batch_size, True, 10)
    test_loader = get_loader(test, vocab, args.batch_size, False, 10)
    return train_loader, valid_loader, test_loader, vocab