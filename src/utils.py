import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import logging
import json
from .models import *

def rmse(real: list, predict: list) -> float:
    '''
    [description]
    RMSE를 계산하는 함수입니다.

    [arguments]
    real : 실제 값입니다.
    predict : 예측 값입니다.

    [return]
    RMSE를 반환합니다.
    '''
    pred = np.array(predict)
    return np.sqrt(np.mean((real-pred) ** 2))


def models_load(args, data):
    '''
    [description]
    입력받은 args 값에 따라 모델을 선택하며, 모델이 존재하지 않을 경우 ValueError를 발생시킵니다.

    [arguments]
    args : argparse로 입력받은 args 값으로 이를 통해 모델을 선택합니다.
    data : data는 data_loader로 처리된 데이터를 의미합니다.
    '''

    if args.model=='FM':
        model = FactorizationMachineModel(args, data).to(args.device)
    elif args.model=='FFM':
        model = FieldAwareFactorizationMachineModel(args, data).to(args.device)
    elif args.model=='NCF':
        model = NeuralCollaborativeFiltering(args, data).to(args.device)
    elif args.model=='WDN':
        model = WideAndDeepModel(args, data).to(args.device)
    elif args.model=='DCN':
        model = DeepCrossNetworkModel(args, data).to(args.device)
    elif args.model=='CNN_FM':
        model = CNN_FM(args, data).to(args.device)
    elif args.model=='DeepCoNN':
        model = DeepCoNN(args, data).to(args.device)
    else:
        raise ValueError('MODEL is not exist : select model in [FM,FFM,NCF,WDN,DCN,CNN_FM,DeepCoNN]')
    return model


class Setting:
    @staticmethod
    def seed_everything(seed):
        '''
        [description]
        seed 값을 고정시키는 함수입니다.

        [arguments]
        seed : seed 값
        '''
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def __init__(self):
        now = time.localtime()
        now_date = time.strftime('%Y%m%d', now)
        now_hour = time.strftime('%X', now)
        save_time = now_date + '_' + now_hour.replace(':', '')
        self.save_time = save_time

    def get_log_path(self, args):
        '''
        [description]
        log file을 저장할 경로를 반환하는 함수입니다.

        [arguments]
        args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.

        [return]
        path : log file을 저장할 경로를 반환합니다.
        이 때, 경로는 log/날짜_시간_모델명/ 입니다.
        '''
        path = f'./log/{self.save_time}_{args.model}/'
        return path

    def get_submit_filename(self, args):
        '''
        [description]
        submit file을 저장할 경로를 반환하는 함수입니다.

        [arguments]
        args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.

        [return]
        filename : submit file을 저장할 경로를 반환합니다.
        이 때, 파일명은 submit/날짜_시간_모델명.csv 입니다.
        '''
        filename = f'./submit/{self.save_time}_{args.model}.csv'
        return filename

    def make_dir(self,path):
        '''
        [description]
        경로가 존재하지 않을 경우 해당 경로를 생성하며, 존재할 경우 pass를 하는 함수입니다.

        [arguments]
        path : 경로

        [return]
        path : 경로
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            pass
        return path


class Logger:
    def __init__(self, args, path):
        """
        [description]
        log file을 생성하는 클래스입니다.

        [arguments]
        args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.
        path : log file을 저장할 경로를 전달받습니다.
        """
        self.args = args
        self.path = path

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('[%(asctime)s] - %(message)s')

        self.file_handler = logging.FileHandler(self.path+'train.log')
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

    def log(self, epoch, train_loss, valid_loss):
        '''
        [description]
        log file에 epoch, train loss, valid loss를 기록하는 함수입니다.
        이 때, log file은 train.log로 저장됩니다.

        [arguments]
        epoch : epoch
        train_loss : train loss
        valid_loss : valid loss
        '''
        message = f'epoch : {epoch}/{self.args.epochs} | train loss : {train_loss:.3f} | valid loss : {valid_loss:.3f}'
        self.logger.info(message)

    def close(self):
        '''
        [description]
        log file을 닫는 함수입니다.
        '''
        self.logger.removeHandler(self.file_handler)
        self.file_handler.close()

    def save_args(self):
        '''
        [description]
        model에 사용된 args를 저장하는 함수입니다.
        이 때, 저장되는 파일명은 model.json으로 저장됩니다.
        '''
        argparse_dict = self.args.__dict__

        with open(f'{self.path}/model.json', 'w') as f:
            json.dump(argparse_dict,f,indent=4)

    def __del__(self):
        self.close()
