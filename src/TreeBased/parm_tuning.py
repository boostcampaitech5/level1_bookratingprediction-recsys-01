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

def prepare_data(data):
    train_data, test_data = data[0], data[1]
    X_train_data, y_train_data = train_data.drop(['user_id', 'isbn', 'rating','book_author'], axis=1), train_data['rating']
    X_test_data, y_test_data = test_data.drop(['user_id', 'isbn', 'rating','book_author'], axis=1), test_data['rating']

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
    #train_data, test_data = data[0], data[1]
    #X_train_data, y_train_data = train_data.drop(['user_id', 'isbn', 'rating','book_author'], axis=1), train_data['rating']
    #X_test_data, y_test_data = test_data.drop(['user_id', 'isbn', 'rating','book_author'], axis=1), test_data['rating']
    
    X_train_data, y_train_data, X_test_data, y_test_data = prepare_data(data)
    
    ######################## Model select
    if args.model in ('catboost'):
        cat_list = [x for x in X_train_data.columns.tolist()]
        model = CatBoostRegressor(cat_features=cat_list, 
                                train_dir = log_path+'catboost_info',
                                task_type = args.device,
                                random_seed= args.seed,
                                bootstrap_type='Poisson'
                                 )
        
    #elif args.model in ('modelname'):
    #    print('')
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

def optuna_obj(trial:Trial, args):
    if args.model in ('catboost'):
        X_train_data, y_train_data, X_test_data, y_test_data = prepare_data(data)
        cat_list = [x for x in X_train_data.columns.tolist()]

        #parameter for optuna
        '''with open('./config/tree_config.json', 'r') as f:
        params = json.load(f)'''

        params = {
        'iterations':trial.suggest_int("iterations", 500, 4000),
        'learning_rate' : trial.suggest_float('learning_rate',0.00001, 0.1),
        'reg_lambda': trial.suggest_float('reg_lambda',1e-5,100),
        'subsample': trial.suggest_float('subsample',0,1),
        'random_strength': trial.suggest_float('random_strength',10,50),
        'depth': trial.suggest_int('depth',3, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',1,30),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',1,15),
        'bagging_temperature' :trial.suggest_float('bagging_temperature', 0.01, 100.00),
        }

        tr_X, val_X, tr_y, val_y = train_test_split(X_train_data,
                                                    y_train_data,
                                                    test_size = 0.2,
                                                    random_state= args.seed,
                                                    shuffle=True
                                                        )
        
        model = CatBoostRegressor(**params,
                              task_type = "GPU",
                              cat_features = cat_list,
                              random_seed= args.seed,
                              bootstrap_type='Poisson',
                              verbose = 100)
        
        model.fit(tr_X,tr_y, use_best_model = True, eval_set = (val_X, val_y))

        val_pred = model.predict(val_X)
        val_pred = val_pred.tolist()
        val_RMSE = rmse(val_y, val_pred)
        return(val_RMSE)



def optuna_search(data, args):

    wandb_kwargs = {"project": "Tree-based models",
                    "entity" : 'recsys01',
                    'name' : args.name,
                    "reinit": True}
    
    wandbc = WeightsAndBiasesCallback(metric_name="RMSE", wandb_kwargs=wandb_kwargs)

    optuna_cbrm = optuna.create_study(direction='minimize', sampler = TPESampler())
    optuna_cbrm.optimize(optuna_obj(args), n_trials = args.n_trials , callbacks =[wandbc])

    for param_name, param_value in optuna_cbrm.best_trial.params.items():
        wandb.run.summary[f(param_name)] = param_value
        wandb.run.summary["best val_RMSE"] = optuna_cbrm.best_trial.value

    wandb.log({"param_importance_chart" : plot_param_importances(optuna_cbrm) ,
               "param_optimization_history" : plot_optimization_history(optuna_cbrm)})

    ####################### Final train&prediction using best model
    print(f"=============best trial value : {optuna_cbrm.best_trial.value}=============")
    print(f"=============best trial parameter : {optuna_cbrm.best_trial.params}=============")

    return optuna_cbrm.best_trial.params