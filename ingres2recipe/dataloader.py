import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from build_vocab import Vocabulary
import Constants
import json
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import os,sys
from nltk.tokenize import TweetTokenizer
nlp = TweetTokenizer()

class Ingres2RecipeDataset(Dataset):
    def __init__(self, vocab, cuisine_vocab, data, feature=None, period="24"):
        self.data = data
        self.vocab = vocab
        self.max_len = 40
        self.cuisine_vocab = cuisine_vocab

    def __getitem__(self, index):
        # 0: recipe_id, 1:instruction/recipe, 2:list of ingres
        recipe_id = self.data[index][0]
        recipe = self.data[index][1]
        ingres = self.data[index][2]
        ingres = " ".join(ingres)
        cuisine = self.data[index][3]
        
        # tokenize recipe
        r_tokens = []
        tokenized_r = nlp.tokenize(recipe)
        r_tokens.extend([self.vocab(token) for i, token in enumerate(tokenized_r) if i<self.max_len])

        # tokenize ingres
        i_tokens = []
        tokenized_i = nlp.tokenize(ingres)
        i_tokens.extend([self.vocab(token) for i, token in enumerate(tokenized_i) if i<self.max_len])

        # cuisine one-hot encoding
        cuisine_id = self.cuisine_vocab[cuisine]
        cuisine_one_hot = [0]*len(self.cuisine_vocab)
        cuisine_one_hot[cuisine_id] = 1
        return r_tokens, i_tokens, cuisine_one_hot, recipe_id

    def collate_fn(self, data):
        recipes, ingres, cuisines, recipe_ids = zip(*data)
        r_lengths = [len(x) for x in recipes]
        i_lengths = [len(x) for x in ingres]

        padded_recipes = [n + [Constants.PAD for _ in range(self.max_len - len(n))] for n in recipes]
        padded_ingres = [n + [Constants.PAD for _ in range(self.max_len - len(n))] for n in ingres]

        recipes = torch.LongTensor(padded_recipes).view(-1, self.max_len)
        ingres = torch.LongTensor(padded_ingres).view(-1, self.max_len)
        cuisines = torch.FloatTensor(cuisines).view(-1, len(self.cuisine_vocab))
        r_lengths = torch.LongTensor(r_lengths).view(-1,1)
        i_lengths = torch.LongTensor(i_lengths).view(-1,1)
        return (recipes, ingres, cuisines, r_lengths, i_lengths), recipe_ids

    def __len__(self):
        return len(self.data)

def get_loader(data, vocab, cuisine_vocab, batch_size, shuffle, num_workers):
    ingres2recipe = Ingres2RecipeDataset(vocab, cuisine_vocab, data)

    data_loader = torch.utils.data.DataLoader(dataset=ingres2recipe,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=ingres2recipe.collate_fn)
    return data_loader

def get_loaders(args):
    with open(f"vocab.pkl",'rb') as f:
        print("----- Loading Vocab -----")
        vocab = pickle.load(f)
        print(f"vocab size: {len(vocab)}")
    cuisine_vocab = {c:i for i, c in enumerate(['italian', 'french', 'british', 'indian', 'southern_us', 'mexican',
       'chinese', 'thai', 'cajun_creole', 'brazilian', 'greek',
       'japanese', 'spanish', 'irish', 'moroccan', 'korean', 'jamaican',
       'vietnamese', 'russian'])}
    print('----- Loading Note -----')

    train = json.load(open('train.json'))
    valid = json.load(open('val.json'))
    test = json.load(open('test.json'))
    
    print("train size", len(train))
    print("val size", len(valid))
    print("test size", len(test))
    print()
    print('----- Building Loaders -----')
    train_loader = get_loader(train, vocab, cuisine_vocab, args.batch_size, True, 10)
    valid_loader = get_loader(valid, vocab, cuisine_vocab, args.batch_size, True, 10)
    test_loader = get_loader(test, vocab, cuisine_vocab, args.batch_size, False, 10)
    return train_loader, valid_loader, test_loader, vocab, cuisine_vocab