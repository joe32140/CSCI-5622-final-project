"""
this code is modified from https://github.com/ksenialearn/bag_of_words_pytorch/blob/master/bag_of_words-master-FINAL.ipynb
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

class DAN(nn.Module):
    """
    BagOfWords classification model
    """
    def __init__(self, vocab_size, emb_dim, feature_len, dropout, is_feature=False, is_text=True):
        """
        @param vocab_size: size of the vocabulary.
        @param emb_dim: size of the word embedding
        """
        super(DAN, self).__init__()
        # pay attention to padding_idx
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        self.linear_r1 = nn.Linear(emb_dim, 200)
        self.linear_r2 = nn.Linear(200, 100)
        self.linear_i1 = nn.Linear(emb_dim, 200)
        self.linear_i2 = nn.Linear(200, 100)

        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(200)

    def forward(self, recipes, ingres, r_length, i_length):
        """
        @param recipes, ingres: matrix of size (batch_size, max_sentence_length). Each row in data represents a
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param r_length, i_length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        r = self.embed(recipes)
        r = self.dropout(r)
        r = torch.sum(r, dim=1)
        r /= r_length.float()
        r = self.linear_r1(r)
        r = self.dropout(r)
        r = self.linear_r2(r)

        ing = self.embed(ingres)
        ing = self.dropout(ing)
        ing = torch.sum(ing, dim=1)
        ing /= i_length.float()
        ing = self.linear_i1(ing)
        ing = self.dropout(ing)
        ing = self.linear_i2(ing)

        return r, ing