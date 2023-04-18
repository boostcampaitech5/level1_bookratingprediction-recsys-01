import numpy as np
import pandas as pd
from src.utils import get_sampler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error

class Stacking:
    '''
    [description]
    스태킹 앙상블을 진행하는 클래스입니다.

    [parameter]
    filenames: 스태킹 앙상블을 진행할 모델의 이름을 리스트 형태로 입력합니다.
    filepath: 스태킹 앙상블을 진행할 모델의 csv 파일이 저장된 경로를 입력합니다.
    '''
    def __init__(self, filenames:str, filepath:str, seed:int, test_size:float):
        self.filenames = filenames
        self.filepath = filepath
        
        self.load_valid_data()
        self.load_output_data()
        self.prepare_train_data(seed=seed, test_size=test_size)

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
        
        
    def prepare_train_data(self, seed, test_size=0.5):
        X_data = np.transpose(self.valid_pred_list)
        y_label = np.array(self.valid_labels)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_data, y_label, test_size=test_size, random_state=seed)
    

    def cv_train(self, X, y):
        skf = StratifiedKFold(n_splits=5)

        for fold, (train_index, test_index) in enumerate(skf.split(X=X, y=y)):
            print(f"----- Fold {fold} -----")
            
            fold_X_train = X[train_index]
            fold_y_train = y[train_index]
            
            fold_X_test = X[test_index]
            fold_y_test = y[test_index]
            
            self.train(fold_X_train, fold_y_train)
            self.test(fold_X_test, fold_y_test)
    

    def train(self, X, y):
        self.lr_final.fit(X, y)
        print(f'Weight: {self.lr_final.coef_}')
        print(f'Bais: {self.lr_final.intercept_}')
        
        
    def test(self, X, y):
        pred = self.lr_final.predict(X)
        loss = rmse(pred, y)
        
        print(f'Train RMSE: {loss}')
        
        
    def get_weights(self):
        return self.lr_final.coef_
    

    def get_bais(self):
        return self.lr_final.intercept_


    def infer(self):
        X_data = np.transpose(self.output_pred_list)
        
        pred = self.lr_final.predict(X_data)
        return pred
        
        
def rmse(a, b):
    return mean_squared_error(a, b) ** 0.5