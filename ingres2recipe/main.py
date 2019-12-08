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
        recipes, ingres, cuisines, r_length, i_length = map(lambda x: x.to(device), batch[0])
        r_embedd, i_embedd = model(recipes, ingres, cuisines, r_length, i_length)

        i_embedd = torch.cat([i_embedd, i_embedd], dim=0)
        r_embedd = torch.cat([r_embedd, torch.flip(r_embedd, [0])], dim=0)
        # backward
        labels = torch.cat([torch.ones(r_length.size()), -1*torch.ones(r_length.size())], dim=0).squeeze().to(device)
        loss = loss_fn(r_embedd, i_embedd, labels)
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
            recipes, ingres, cuisines, r_length, i_length = map(lambda x: x.to(device), batch[0])
            r_embedd, i_embedd = model(recipes, ingres, cuisines, r_length, i_length)

            i_embedd = torch.cat([i_embedd, i_embedd], dim=0)
            r_embedd = torch.cat([r_embedd, torch.flip(r_embedd, [0])], dim=0)
            # backward
            labels = torch.cat([torch.ones(r_length.size()), -1*torch.ones(r_length.size())], dim=0).squeeze().to(device)
            loss = loss_fn(r_embedd, i_embedd, labels)
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
    query_embedds = []
    target_embedds = []
    recipe_ids = []
    with torch.no_grad():
        for batch in tqdm(
                test_data, mininterval=2,
                desc='  - (Validation) ', leave=False):
            # prepare data
            recipes, ingres, cuisines, r_length, i_length = map(lambda x: x.to(device), batch)
            r_embedd, i_embedd = model(recipes, ingres, cuisines, r_length, i_length)
            
            query_embedds.append(i_embedd)
            target_embedds.append(r_embedd)
    query_embedds = torch.cat(query_embedds, dim=0)
    target_embedds = torch.cat(target_embedds, dim=0)

    ranking(query_embedds, target_embedds, recipe_ids)

def ranking(query_embedds, target_embedds, img_ids):                                                                                                          
    """                                                                                                                                                       
    @ param query_embedds = (n, d)                                                                                                                            
    @ param target_embedds = (n, d)                                                                                                                           
    @ param img_ids = (n,)                                                                                                                                    
    """
    cos_sim = torch.mm(query_embedds,target_embedds.T)/ \
        torch.mm(query_embedds.norm(2, dim=1, keepdim=True),                                                                                          
                    target_embedds.norm(2, dim=1, keepdim=True).T)                                                                                        
    _, idx = torch.topk(cos_sim, len(query_embedds)//100, dim=1)                                                                                              
    top20 = idx.cpu().numpy()                                                                                                                                 
    img_ids = np.array(img_ids)                                                                                                                               
    count = 0                                                                                                                                                 
    with open('answer.csv', 'w') as f:                                                                                                                        
        f.write("Descritpion_ID,Top_20_Image_IDs\n")                                                                                                          
        for i, img_id in enumerate(img_ids):                                                                                                                  
            top_imgs = img_ids[top20[i]]                                                                                                                      
            top_imgs_str = " ".join(list(top_imgs))                                                                                                           
            text_id = img_id.split(".")[0]+".txt"                                                                                                             
            f.write(text_id+","+top_imgs_str+"\n")                                                                                                            
            if img_id in list(top_imgs):                                                                                                                      
                count+=1                                                                                                                                      
        print("count", count)

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
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

        scheduler.step(train_loss)

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
    training_data, validation_data, test_data, vocab, cuisine_label= dataloader.get_loaders(opt)

    #========= Preparing Model =========#
    print(opt)

    device = torch.device(f'cuda:{opt.device}' if opt.cuda else 'cpu')

    dan = Ingres2Recipe(len(vocab), len(cuisine_label), opt.embedding_size, opt.dropout).to(device)
    optimizer = optim.Adam(
            dan.parameters(),
            betas=(0.9, 0.98), eps=1e-09, lr=0.003)
    loss_fn = nn.CosineEmbeddingLoss(margin=0.2)
    if not opt.test_mode:
        train(dan, training_data, validation_data, optimizer, loss_fn, device ,opt)

    model_name = 'model.chkpt'
    
    checkpoint = torch.load(f"./models/{model_name}", map_location=device)
    dan.load_state_dict(checkpoint['model'])
    test(dan, validation_data, loss_fn, device, opt)
    #predict_prob(dan, test_data, loss_fn, device, opt)
if __name__ == '__main__':
    main()