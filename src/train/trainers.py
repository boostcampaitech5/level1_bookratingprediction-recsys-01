import os
import torch
import wandb
import numpy as np
import pandas as pd
from src.data import context_data_load, context_data_split, context_data_loader
from src.data import dl_data_load, dl_data_split, dl_data_loader
from src.data import image_data_load, image_data_split, image_data_loader
from src.data import text_data_load, text_data_split, text_data_loader
from src.data import image_context_data_load, image_context_data_split, image_context_data_loader
from src.data import image_text_data_load, image_text_data_split, image_text_data_loader
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
from src.utils import get_sampler, models_load, loss_fn_load, optimizer_load

class TrainerBase():
    
    
    def __init__(self, args, logger, setting):
        self.load_data(args)
        self.logger = logger
        self.setting = setting
        
        
    def load_data(self, args):
        print(f'--------------- {args.model} Load Data ---------------')
        if args.model in ('FM', 'FFM', 'FFDCN'):
            self.data = context_data_load(args)
        elif args.model in ('NCF', 'WDN', 'DCN'):
            self.data = dl_data_load(args)
        elif args.model == 'CNN_FM':
            if args.cnn_feed_context:
                self.data = image_context_data_load(args)
            else:
                self.data = image_data_load(args)
        elif args.model == 'DeepCoNN':
            import nltk
            nltk.download('punkt')
            self.data = text_data_load(args)
        elif args.model == 'DeepCoNN_CNN':
            import nltk
            nltk.download('punkt')
            self.data = image_text_data_load(args)
            
            
class Trainer(TrainerBase):
  
  
    def __init__(self, args, logger, setting):
        super().__init__(args, logger, setting)
        self.prepare_dataloader(args)
        
    
    def prepare_dataloader(self, args):
        print(f'--------------- {args.model} Train/Valid Split ---------------')
        data = split_data(args, self.data)
        
        self.data = data
        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.test_dataloader = data['test_dataloader']
    
    
    def train(self, args):
        # init model, loss_fn, optimizer
        model = models_load(args, self.data)
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

            for idx, data in enumerate(self.train_dataloader):
                x, y = get_x_y(args, data)
                
                y_hat = model(x)
                loss = loss_fn(y.float(), y_hat)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch +=1
                
            valid_loss = valid(args, model, self.valid_dataloader, loss_fn)
            
            # logging
            print(f'Epoch: {epoch+1}, Train_loss: {total_loss/batch:.3f}, valid_loss: {valid_loss:.3f}')
            self.logger.log(epoch=epoch+1, train_loss=total_loss/batch, valid_loss=valid_loss)
            wandb.log({'epoch': epoch, 'tra_loss': total_loss/batch, 'val_loss': valid_loss})
            
            if minimum_loss > valid_loss:
                minimum_loss = valid_loss
                os.makedirs(args.saved_model_path, exist_ok=True)
                torch.save(model.state_dict(), f'{args.saved_model_path}/{self.setting.save_time}_{args.model}_model.pt')

            if valid_loss > best_loss:
                patient_check += 1
                if patient_check >= patient_limit:
                    break
            else:
                best_loss = valid_loss
                patient_check = 0
                
                
        self.traind_model = model
            
        print(f"score: {minimum_loss:4f}")
        return self.traind_model, minimum_loss
    
    
    def test(self, args):
        return predict(args, self.traind_model, self.test_dataloader)
    
    
    def infer(self, args):
        valid_predicts = predict(args, self.traind_model, self.valid_dataloader)
        return infer(valid_predicts, self.data)


class CVTrainer(TrainerBase):
    
    
    def __init__(self, args, logger, setting):
        super().__init__(args, logger, setting)
        self.prepare_dataloader(args)
        
    
    def prepare_dataloader(self, args):
        print(f'--------------- {args.model} Load Dataset ---------------')
        data = split_data(args, self.data)
        
        self.data = data
        self.whole_dataset = data['train_dataloader'].dataset
        self.test_dataloader = data['test_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        
        if hasattr(self.whole_dataset, 'label'):
            self.y_rating = self.whole_dataset.label
        else:
            self.y_rating = self.whole_dataset.tensors[1]
        

    def train(self, args):
        
        kf = KFold(n_splits= 5, shuffle=True, random_state=args.seed)
        model_list = []
        cv_score = 0
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(self.whole_dataset)):
            train_dataset = Subset(self.whole_dataset, train_idx)
            valid_dataset = Subset(self.whole_dataset, valid_idx)
            
            y_train = self.y_rating[train_idx]

            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, sampler=get_sampler(args, train_dataset, y_train))
            valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
            
            model, minimum_loss = self.train_fold(args, train_dataloader, valid_dataloader, fold=fold)
            model_list.append(model)
            
            cv_score += minimum_loss/5
            
        print(f"cv_score: {cv_score:4f}")
        wandb.log({'cv_score': cv_score})
        
        self.trained_model_list = model_list
        
        return model_list, cv_score
    
    
    def train_fold(self, args, train_dataloader, valid_dataloader, fold):
        # init model, loss_fn, optimizer
        model = models_load(args, self.data)
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

            for idx, data in enumerate(train_dataloader):
                x, y = get_x_y(args, data)
                
                y_hat = model(x)
                loss = loss_fn(y.float(), y_hat)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch +=1
                
            valid_loss = valid(args, model, valid_dataloader, loss_fn)
            
            # logging by fold
            print(f'Fold: {fold+1} Epoch: {epoch+1}, Train_loss: {total_loss/batch:.3f}, valid_loss: {valid_loss:.3f}')
            self.logger.log(epoch=epoch+1, train_loss=total_loss/batch, valid_loss=valid_loss)
            wandb.log({'fepoch': epoch, 'ftra_loss': total_loss/batch, 'fval_loss': valid_loss})
        
            if minimum_loss > valid_loss:
                minimum_loss = valid_loss
                os.makedirs(args.saved_model_path, exist_ok=True)
                torch.save(model.state_dict(), f'{args.saved_model_path}/{self.setting.save_time}_{args.model}_model{fold}.pt')

            if valid_loss > best_loss:
                patient_check += 1
                if patient_check >= patient_limit:
                    break
            else:
                best_loss = valid_loss
                patient_check = 0
                
        print(f"score: {minimum_loss:4f}")
        return model, minimum_loss
    
    
    def test(self, args):
        return predict(args, self.trained_model_list[0], self.test_dataloader)
    
    
    def oof_test(self, args):
        return self.oof_predict(args, self.test_dataloader)
    
    
    def infer(self, args):
        valid_predicts = predict(args, self.trained_model_list[0], self.valid_dataloader)
        return infer(valid_predicts, self.data)
    
    
    def oof_infer(self, args):
        valid_predicts = self.oof_predict(args, self.valid_dataloader)
        return infer(valid_predicts, self.data)
    
    
    def oof_predict(self, args, dataloader):
        predicts_list = []
        for fold, model in enumerate(self.trained_model_list):
            predicts_fold = np.array(predict(args, model, dataloader))
            predicts_list.append(predicts_fold)

        predicts = np.sum(predicts_list, axis=0) / len(self.trained_model_list)
        return list(predicts)
    

def infer(valid_predicts, data):
    # if args.use_best_model == True:
        # model.load_state_dict(torch.load(f'./saved_models/{setting.save_time}_{args.model}_model{fold}.pt'))
    # else:
    #     pass
    # model.eval()
    
    user_id = data['X_valid']['user_id'].map(data['idx2user']).reset_index(drop=True)
    isbn = data['X_valid']['isbn'].map(data['idx2isbn']).reset_index(drop=True)
    rating = data['y_valid'].reset_index(drop=True)
    
    pred = pd.Series(valid_predicts, name='pred')
    return pd.concat([user_id, isbn, rating, pred], axis=1)


def predict(args, model, dataloader):
    predicts = list()
    
    # model.load_state_dict(torch.load(f'./saved_models/{self.setting.save_time}_{args.model}_model.pt'))
    # model.eval()

    for idx, data in enumerate(dataloader):
        x, _ = get_x_y(args, data)
        y_hat = model(x)
        predicts.extend(y_hat.tolist())
    return predicts

    
def valid(args, model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    batch = 0

    for idx, data in enumerate(dataloader):
        x, y = get_x_y(args, data)
        
        y_hat = model(x)
        loss = loss_fn(y.float(), y_hat)
        
        total_loss += loss.item()
        batch +=1
        
    valid_loss = total_loss/batch
    return valid_loss


def split_data(args, data):
    if args.model in ('FM', 'FFM', 'FFDCN'):
        data = context_data_split(args, data)
        data = context_data_loader(args, data)

    elif args.model in ('NCF', 'WDN', 'DCN'):
        data = dl_data_split(args, data)
        data = dl_data_loader(args, data)

    elif args.model=='CNN_FM':
        if args.cnn_feed_context:
            data = image_context_data_split(args, data)
            data = image_context_data_loader(args, data)
        else:
            data = image_data_split(args, data)
            data = image_data_loader(args, data)

    elif args.model=='DeepCoNN':
        data = text_data_split(args, data)
        data = text_data_loader(args, data)
        
    elif args.model=='DeepCoNN_CNN':
        data = image_text_data_split(args, data)
        data = image_text_data_loader(args, data)
    
    return data


def get_x_y(args, data):
    if args.model == 'CNN_FM':
        x, y = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
    elif args.model == 'DeepCoNN':
        x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
    elif args.model == 'DeepCoNN_CNN':
        x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
    else:
        if len(data) >= 2:
            x, y = data[0].to(args.device), data[1].to(args.device)
        else:
            x, y = data[0].to(args.device), None

    return x, y