import argparse
import math
import time
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
from model import Ingres2Recipe
import dataloader
from build_vocab import Vocabulary
from sklearn.metrics import roc_auc_score
import pandas as pd
def train_epoch(model, training_data, optimizer, loss_fn, device, opt):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss = 0
    count = 0
    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        optimizer.zero_grad()
        # prepare data
        recipes, ingres, r_length, i_length = map(lambda x: x.to(device), batch)
        r_embedd, i_embedd = model(recipes, ingres, r_length, i_length)
        # backward
        loss = loss_fn(r_embedd, i_embedd, torch.ones(r_length.size()))
        loss.backward()
        optimizer.step()
        # note keeping
        total_loss += loss.item()

        count +=1
        #if count%10==0:
        #    print(f"Loss: {loss.item()}")
        #print("===============================================\n")

    loss = total_loss/count
    return loss

def eval_epoch(model, validation_data, loss_fn, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    count=0
    total_loss = 0
    mortality_all = []
    pred_all = []
    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):
            # prepare data
            recipes, ingres, r_length, i_length = map(lambda x: x.to(device), batch)
            r_embedd, i_embedd = model(recipes, ingres, r_length, i_length)
            # backward
            loss = loss_fn(r_embedd, i_embedd, torch.ones(r_length.size()))
            # note keeping
            total_loss += loss.item()
            count +=1
    loss = total_loss/count
    print("Loss:", loss)
    return loss

def test(model, test_data, loss_fn, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    count=0
    total_loss = 0
    mortality_all = []
    pred_all = []
    with torch.no_grad():
        for batch in tqdm(
                test_data, mininterval=2,
                desc='  - (Validation) ', leave=False):
            # prepare data
            recipes, ingres, r_length, i_length = map(lambda x: x.to(device), batch)
            r_embedd, i_embedd = model(recipes, ingres, r_length, i_length)

            # backward
            loss = loss_fn(r_embedd, i_embedd)
            # note keeping
            total_loss += loss.item()
            count +=1
    loss_per_word = total_loss/count
    print("----- Test Result -----")
    print("Loss:", loss_per_word)

def train(model, training_data, validation_data, optimizer, loss_fn, device, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None
    log_dir = opt.log

    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        print(f"Scheduled Learning Rate:{lr}")
    if opt.log:
        log_train_file = log_dir + 'train.log'
        log_valid_file = log_dir + 'valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            #log_tf.write('epoch,loss,ppl,accuracy\n')
            #log_vf.write('epoch,loss,ppl,accuracy\n')
            log_tf.write('epoch,loss\n')
            log_vf.write('epoch,loss\n')

    valid_losses = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss  = train_epoch(
            model, training_data, optimizer, loss_fn, device, opt=opt)
        #print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
        #      'elapse: {elapse:3.3f} min'.format(
        #          ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
        #          elapse=(time.time()-start)/60))
        print('  - (Training)   loss: {loss: 8.5f}, '\
              'elapse: {elapse:3.3f} min'.format(
                  loss=train_loss,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss = eval_epoch(model, validation_data, loss_fn, device, opt)
        # print('  - (Validation) ppl: {ppl: 8.5f}, roc_auc: {accu:3.3f} %, '\
        #         'elapse: {elapse:3.3f} min'.format(
        #             ppl=math.exp(min(valid_loss, 100)), accu=100*valid_roc_auc,
        #             elapse=(time.time()-start)/60))
        print('  - (Validation) loss: {loss: 8.5f}, '\
               'elapse: {elapse:3.3f} min'.format(
                   loss=valid_loss,
                   elapse=(time.time()-start)/60))

        valid_losses += [valid_loss]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            save_dir = "./models/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if opt.save_mode == 'all':
                model_name = save_dir + opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_loss)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = save_dir+'model.chkpt'
                if valid_loss <= min(valid_losses):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                #log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                #    epoch=epoch_i, loss=train_loss,
                #    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                # log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                #     epoch=epoch_i, loss=valid_loss,
                #     ppl=math.exp(min(valid_loss, 100)), accu=100*valid_roc_auc))
                log_tf.write('{epoch},{loss: 8.5f}\n'.format(
                    epoch=epoch_i, loss=train_loss))
                log_vf.write('{epoch},{loss: 8.5f}\n'.format(
                   epoch=epoch_i, loss=valid_loss,))

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=32)

    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-embedding_size', type=float, default=300)

    parser.add_argument('-log', type=str, default="./log/")
    parser.add_argument('-save_model', default=True)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-test_mode', action='store_true', default=False)

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-device', type=str, default='0')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    #========= Loading Dataset =========#
    torch.manual_seed(1234)
    training_data, validation_data, test_data, vocab= dataloader.get_loaders(opt)

    #========= Preparing Model =========#
    print(opt)

    device = torch.device(f'cuda:{opt.device}' if opt.cuda else 'cpu')

    dan = Ingres2Recipe(len(vocab), opt.embedding_size, opt.dropout).to(device)
    optimizer = optim.Adam(
            dan.parameters(),
            betas=(0.9, 0.98), eps=1e-09, lr=0.0001)
    loss_fn = nn.CosineEmbeddingLoss()
    if not opt.test_mode:
        train(dan, training_data, validation_data, optimizer, loss_fn, device ,opt)

    model_name = 'model.chkpt'
    
    checkpoint = torch.load(f"./models/{model_name}", map_location=device)
    dan.load_state_dict(checkpoint['model'])
    test(dan, validation_data, loss_fn, device, opt)
    #predict_prob(dan, test_data, loss_fn, device, opt)
if __name__ == '__main__':
    main()