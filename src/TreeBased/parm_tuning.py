import time
import argparse
import wandb
import os
import dotenv
import config
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from src.data import catboost_Data
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor

import warnings

def prepare_data(data):
    train_data, test_data = data[0], data[1]
    X_train_data, y_train_data = train_data.drop(['rating'], axis=1), train_data['rating']
    X_test_data, y_test_data = test_data.drop(['rating'], axis=1), test_data['rating']

    return X_train_data, y_train_data, X_test_data, y_test_data


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



def grid_search(data, log_path, args):
    ######################## setting data
    X_train_data, y_train_data, X_test_data, y_test_data = prepare_data(data)
    cat_list = [x for x in X_train_data.columns.tolist()]
    print(cat_list)

    
    if args.model == 'lgbm':
        X_test_data = X_test_data.astype('category')
        X_train_data[cat_list] = X_train_data[cat_list].astype('category')
    
    ######################## Model select
    if args.model in ('catboost'):
        model = CatBoostRegressor(cat_features=cat_list, 
                                train_dir = log_path+'catboost_info',
                                task_type = args.device,
                                random_seed= args.seed,
                                # bootstrap_type='Poisson'
                                 )
        
    elif args.model in ('lgbm'):
        model = LGBMRegressor(cat_feature = cat_list)
    else:
        pass
    
    #ex_param.txt에서 원하는 case에 맞는 형태를 작성+복사한 후, tree_config.jason에 붙여넣어주세요.
    with open('./config/tree_config.json', 'r') as f:
        params = json.load(f)
    wandb.save('./config/tree_config.json')
    
    cv = StratifiedKFold(n_splits= args.n_fold, shuffle=args.data_shuffle, random_state=args.seed)
    grid = GridSearchCV(estimator=model, cv = cv, param_grid=params)
    
    grid.fit(X_train_data, y_train_data)

    wandb.log({'best_valid_performance': grid.best_score_})
    
    print(f"====={args.n_fold}-Fold CV result - valid RMSE : {grid.best_score_}=====")
    print(f"====={args.n_fold}-Fold CV result - best parameters : {grid.best_params_}=====")
    
    saving_param = grid.best_params_
    f = "best_{}".format
    for s_parm in saving_param.items():
        wandb.run.summary[f(s_parm[0])] = s_parm[1]
        
    # saving best parameter
    with open(log_path+'best_parm.json', 'w') as f:
        json.dump(saving_param, f)
        
    wandb.save(log_path+'best_parm.json')
    
    return grid.best_estimator_
