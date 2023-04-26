import numpy as np
import pandas as pd
from src.utils import get_sampler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import mean_squared_error

class StackingBase:
    '''
    [description]
    스태킹 앙상블을 진행하는 클래스입니다.

    [parameter]
    filenames: 스태킹 앙상블을 진행할 모델의 이름을 리스트 형태로 입력합니다.
    filepath: 스태킹 앙상블을 진행할 모델의 csv 파일이 저장된 경로를 입력합니다.
    '''
    def __init__(self, filenames:str, filepath:str):
        self.filenames = filenames
        self.filepath = filepath
        
        self.load_valid_data()
        self.load_output_data()

        # stacking
        self.lr_final = LinearRegression()
        
        
    def load_valid_data(self):
        '''
        각 모델의 valid 데이터에 대한 예측 데이터를 불러옵니다.
        '''
        valid_path = [self.filepath+filename+'_valid.csv' for filename in self.filenames]
        self.valid_frame = pd.read_csv(valid_path[0])#.drop('pred',axis=1)
        self.valid_labels = self.valid_frame['rating'].to_list()
        self.valid_pred_list = []
        
        for path in valid_path:
            self.valid_pred_list.append(pd.read_csv(path)['pred'].to_list())
            
            
    def load_output_data(self):
        '''
        각 모델의 test 데이터에 대한 예측 데이터를 불러옵니다.
        '''
        output_path = [self.filepath+filename+'.csv' for filename in self.filenames]
        self.output_frame = pd.read_csv(output_path[0]).drop('rating',axis=1)
        self.output_pred_list = []
        
        for path in output_path:
            self.output_pred_list.append(pd.read_csv(path)['rating'].to_list())
      
        
class Stacking(StackingBase):


    def __init__(self, filenames: str, filepath: str, seed: int, test_size: float):
        super().__init__(filenames, filepath)
        
        self.prepare_train_data(seed, test_size)
        

    def prepare_train_data(self, seed, test_size=0.5):
        X_data = np.transpose(self.valid_pred_list)
        y_label = np.array(self.valid_labels)
        
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X_data, y_label, test_size=test_size, random_state=seed)


    def train(self, verbose=True):
        self.lr_final.fit(self.X_train, self.y_train)
        
        if verbose:
            print(f'Weight: {self.lr_final.coef_}')
            print(f'Bais: {self.lr_final.intercept_}')
        
        
    def valid(self, verbose=True):
        pred = self.lr_final.predict(self.X_valid)
        loss = rmse(pred, self.y_valid)
        
        if verbose:
            print(f'RMSE: {loss}')
        
        
    def get_weights(self):
        return self.lr_final.coef_
    

    def get_bais(self):
        return self.lr_final.intercept_


    def infer(self):
        X_data = np.transpose(self.output_pred_list)
        
        pred = self.lr_final.predict(X_data)
        return pred
    
    
    def get_identity(self):
        weights = self.get_weights()
        weight_info = '-'.join([str(w)[:4] for w in weights])
        files_title = '-'.join(self.filenames)
        
        return f'stacked-sw-{weight_info}-{files_title}'


class OofStacking(StackingBase):
    
    
    def __init__(self, filenames: str, filepath: str):
        super().__init__(filenames, filepath)
        
    
    def train(self):
        X = np.transpose(self.valid_pred_list)
        y = np.array(self.valid_labels)
        
        self.cv_result = cross_validate(self.lr_final, X, y, cv=5, return_estimator=True)
        
        
    def valid(self, verbose=True):
        X = np.transpose(self.valid_pred_list)
        y = np.array(self.valid_labels)
        
        rmses = []
        for model in self.cv_result['estimator']:
            pred = model.predict(X)
            rmses.append(rmse(pred, y))
        rmses = np.array(rmses)
        
        if verbose:
            print('----- Cross validation valid  -----')
            print(f'valid RMSE mean: {rmses.mean()}')
        
        return rmses
    
    
    def infer(self):
        X = np.transpose(self.output_pred_list)
        
        preds = []
        for model in self.cv_result['estimator']:
            preds.append(model.predict(X))
        preds = np.array(preds)
        
        y_hat_oof = preds.mean(axis=0)
        
        return y_hat_oof
    
    
    def get_identity(self):
        files_title = '-'.join(self.filenames)
        
        return f'stacked-cv-oof-{files_title}'

        
def rmse(a, b):
    return mean_squared_error(a, b) ** 0.5