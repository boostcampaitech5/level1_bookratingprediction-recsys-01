import time
import argparse
import wandb
import os
import dotenv
import config
import json
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.visualization import plot_param_importances, plot_optimization_history
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from src.data import catboost_Data
from catboost import CatBoostRegressor, Pool

import warnings


def grid_search(data, log_path, args):
    ######################## setting data
    train_data, test_data = data[0], data[1]
    X_train_data, y_train_data = train_data.drop(['user_id', 'isbn', 'rating','book_author'], axis=1), train_data['rating']
    X_test_data, y_test_data = test_data.drop(['user_id', 'isbn', 'rating','book_author'], axis=1), test_data['rating']
    
    
    ######################## Model select
    if args.model in ('catboost'):
        cat_list = [x for x in X_train_data.columns.tolist()]
        model = CatBoostRegressor(cat_features=cat_list, 
                                train_dir = log_path+'catboost_info',
                                task_type = args.device,
                                random_seed= args.seed,
                                bootstrap_type='Poisson'
                                 )
        
    elif args.model in ('modelname'):
        print('kkwang izi long')
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
    
    return grid.best_estimator_



def optuna_search():
    print('haha')