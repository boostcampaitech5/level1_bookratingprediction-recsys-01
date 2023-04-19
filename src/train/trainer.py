import os
import wandb
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
from src.utils import get_sampler, models_load, loss_fn_load, optimizer_load


def train(args, dataloader, logger, setting, need_log=True, fold=""):
    # init model, loss_fn, optimizer
    model = models_load(args, dataloader)
    loss_fn = loss_fn_load(args)
    optimizer = optimizer_load(args, model)

    minimum_loss = 999999999
    for epoch in range(args.epochs):
        #for early stopping
        best_loss = 10 ** 2 # loss 초기값
        patient_limit = args.patient_limit # 3번의 epoch까지 허용
        patient_check = 0 # 연속적으로 개선되지 않은 epoch의 수
        
        model.train()
        total_loss = 0
        batch = 0

        for idx, data in enumerate(dataloader['train_dataloader']):
            if args.model == 'CNN_FM':
                x, y = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
            elif args.model == 'DeepCoNN':
                x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
            elif args.model == 'DeepCoNN_CNN':
                x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
            else:
                x, y = data[0].to(args.device), data[1].to(args.device)
            y_hat = model(x)
            loss = loss_fn(y.float(), y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch +=1
        valid_loss = valid(args, model, dataloader, loss_fn)
        if need_log:
            print(f'Epoch: {epoch+1}, Train_loss: {total_loss/batch:.3f}, valid_loss: {valid_loss:.3f}')
            logger.log(epoch=epoch+1, train_loss=total_loss/batch, valid_loss=valid_loss)
            wandb.log({'epoch': epoch, 'tra_loss': total_loss/batch, 'val_loss': valid_loss})
        else:
            print(f'Fold: {int(fold)+1} Epoch: {epoch+1}, Train_loss: {total_loss/batch:.3f}, valid_loss: {valid_loss:.3f}')
            logger.log(epoch=epoch+1, train_loss=total_loss/batch, valid_loss=valid_loss)
            wandb.log({'fepoch': epoch, 'ftra_loss': total_loss/batch, 'fval_loss': valid_loss})
        
        if minimum_loss > valid_loss:
            minimum_loss = valid_loss
            os.makedirs(args.saved_model_path, exist_ok=True)
            torch.save(model.state_dict(), f'{args.saved_model_path}/{setting.save_time}_{args.model}_model{fold}.pt')

        if valid_loss > best_loss:
            patient_check += 1
            if patient_check >= patient_limit:
                break
        else:
            best_loss = valid_loss
            patient_check = 0     
    print(f"score: {minimum_loss:4f}")
    return model, minimum_loss

def cv_train(args, dataloader, logger, setting):
    dataset = dataloader['whole_dataset']
    kf = KFold(n_splits= 5, shuffle=True, random_state=args.seed)

    model_list = []
    cv_score = 0
    for fold, (train_idx, valid_idx) in enumerate(kf.split(dataset)):
        train_dataset = Subset(dataset, train_idx)
        valid_dataset = Subset(dataset, valid_idx)
        y_train = dataloader['train']['rating'][train_idx]

        dataloader['train_dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, sampler=get_sampler(args, train_dataset, y_train.values))
        dataloader['valid_dataloader'] = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
        model, minimum_loss = train(args, dataloader, logger, setting, need_log=False, fold=str(fold))
        model_list.append(model)
        cv_score += minimum_loss/5
    print(f"cv_score: {cv_score:4f}")
    wandb.log({'cv_score': cv_score})
    return model_list, cv_score

def valid(args, model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    batch = 0

    for idx, data in enumerate(dataloader['valid_dataloader']):
        if args.model == 'CNN_FM':
            x, y = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
        elif args.model == 'DeepCoNN':
            x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
        elif args.model == 'DeepCoNN_CNN':
            x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
        else:
            x, y = data[0].to(args.device), data[1].to(args.device)
        y_hat = model(x)
        loss = loss_fn(y.float(), y_hat)
        total_loss += loss.item()
        batch +=1
    valid_loss = total_loss/batch
    return valid_loss

def test(args, model, dataloader, setting, fold=""):
    predicts = list()
    model.load_state_dict(torch.load(f'./saved_models/{setting.save_time}_{args.model}_model{fold}.pt'))

    model.eval()

    for idx, data in enumerate(dataloader['test_dataloader']):
        if args.model == 'CNN_FM':
            x, _ = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
        elif args.model == 'DeepCoNN':
            x, _ = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
        elif args.model == 'DeepCoNN_CNN':
            x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
        else:
            x = data[0].to(args.device)
        y_hat = model(x)
        predicts.extend(y_hat.tolist())
    return predicts

def oof_test(args, model_list, dataloader, setting):
    predicts_list = []
    for fold, model in enumerate(model_list):
        predicts_fold = np.array(test(args, model, dataloader, setting, str(fold)))
        predicts_list.append(predicts_fold)

    predicts = np.sum(predicts_list, axis=0) / len(model_list)
    return list(predicts)

def infer(args, model, dataloader, setting, fold=""):
    predicts = list()
    if args.use_best_model == True:
        model.load_state_dict(torch.load(f'./saved_models/{setting.save_time}_{args.model}_model{fold}.pt'))
    else:
        pass
    model.eval()

    for idx, data in enumerate(dataloader['valid_dataloader']):
        if args.model == 'CNN_FM':
            x, _ = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
        elif args.model == 'DeepCoNN':
            x, _ = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
        elif args.model == 'DeepCoNN_CNN':
            x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
        else:
            x = data[0].to(args.device)
        y_hat = model(x)
        predicts.extend(y_hat.tolist())
    
    user_id = dataloader['X_valid']['user_id'].map(dataloader['idx2user']).reset_index(drop=True)
    isbn = dataloader['X_valid']['isbn'].map(dataloader['idx2isbn']).reset_index(drop=True)
    rating = dataloader['y_valid'].reset_index(drop=True)
    pred = pd.Series(predicts, name='pred')
    return pd.concat([user_id, isbn, rating, pred], axis=1)
