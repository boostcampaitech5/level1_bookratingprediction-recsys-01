import os
import tqdm
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim import SGD, Adam


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6
    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y)+self.eps)
        return loss


def train(args, model, dataloader, logger, setting):
    minimum_loss = 999999999
    if args.loss_fn == 'MSE':
        loss_fn = MSELoss()
    elif args.loss_fn == 'RMSE':
        loss_fn = RMSELoss()
    else:
        pass
    if args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'ADAM':
        optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        pass

    for epoch in tqdm.tqdm(range(args.epochs)):
        model.train()
        total_loss = 0
        batch = 0

        for idx, data in enumerate(dataloader['train_dataloader']):
            if args.model == 'CNN_FM':
                x, y = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
            elif args.model == 'DeepCoNN':
                x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
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
        print(f'Epoch: {epoch+1}, Train_loss: {total_loss/batch:.3f}, valid_loss: {valid_loss:.3f}')
        logger.log(epoch=epoch+1, train_loss=total_loss/batch, valid_loss=valid_loss)
        if minimum_loss > valid_loss:
            minimum_loss = valid_loss
            os.makedirs(args.saved_model_path, exist_ok=True)
            torch.save(model.state_dict(), f'{args.saved_model_path}/{setting.save_time}_{args.model}_model.pt')
    logger.close()
    return model


def valid(args, model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    batch = 0

    for idx, data in enumerate(dataloader['valid_dataloader']):
        if args.model == 'CNN_FM':
            x, y = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
        elif args.model == 'DeepCoNN':
            x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
        else:
            x, y = data[0].to(args.device), data[1].to(args.device)
        y_hat = model(x)
        loss = loss_fn(y.float(), y_hat)
        total_loss += loss.item()
        batch +=1
    valid_loss = total_loss/batch
    return valid_loss


def test(args, model, dataloader, setting):
    predicts = list()
    if args.use_best_model == True:
        model.load_state_dict(torch.load(f'./saved_models/{setting.save_time}_{args.model}_model.pt'))
    else:
        pass
    model.eval()

    for idx, data in enumerate(dataloader['test_dataloader']):
        if args.model == 'CNN_FM':
            x, _ = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
        elif args.model == 'DeepCoNN':
            x, _ = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
        else:
            x = data[0].to(args.device)
        y_hat = model(x)
        predicts.extend(y_hat.tolist())
    return predicts
