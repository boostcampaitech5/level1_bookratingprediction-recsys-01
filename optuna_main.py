import time
import argparse
import wandb
import os
import dotenv
import config
import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.visualization import plot_param_importances, plot_optimization_history
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from src.utils import Logger, Setting, models_load, get_timestamp
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor 
from src.data import catboost_Data

import warnings

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

def CB_optuna(trial:Trial):
    if args.model == 'catboost':
        params = {
            'iterations':trial.suggest_int("iterations", 50, 2000),
            'learning_rate' : trial.suggest_float('learning_rate',1e-8, 0.1),
            'reg_lambda': trial.suggest_float('reg_lambda',1e-5,100),
            'subsample': trial.suggest_float('subsample',0,1),
            'random_strength': trial.suggest_float('random_strength',10,50),
            'depth': trial.suggest_int('depth',1, 15),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',1,30),
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',1,15),
            'bagging_temperature' :trial.suggest_float('bagging_temperature', 0.01, 100.00)
            }
    
    elif args.model == 'lgbm':
        params = {
            'objective': 'regression',
            'verbosity': -1,
            'metric': 'rmse', 
            'max_depth': trial.suggest_int('max_depth',3, 15),
            'learning_rate' : trial.suggest_float('learning_rate',1e-8, 0.1),
            'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.4, 1)
            }

    ######################## Load Dataset
    train, test = catboost_Data(args)

    X_train, y_train = train.drop(['rating','book_author'], axis=1), train['rating']
    X_test, y_test = test.drop(['rating','book_author'], axis=1), test['rating']
    
    cat_list = [x for x in X_train.columns.tolist()]

    if args.model == 'lgbm':
        X_test = X_test.astype('category')
        X_train[cat_list] = X_train[cat_list].astype('category')

    '''if args.model == 'lgbm':
        train_lgbm = lightgbm.Dataset(X_train, label=y_train, categorical_feature=cat_list)'''

    

    tr_X, val_X, tr_y, val_y = train_test_split(X_train,
                                                    y_train,
                                                    test_size = 0.2,
                                                    random_state= args.seed,
                                                    shuffle=True
                                                        )
    
    if args.model == 'catboost':
        model = CatBoostRegressor(**params,
                                  task_type = "GPU",
                                  cat_features = cat_list,
                                  random_seed= args.seed,
                                  bootstrap_type='Poisson',
                                  verbose = 100)
        model.fit(tr_X,tr_y, use_best_model = True, eval_set = (val_X, val_y))
    
    elif args.model == 'lgbm':
        model = LGBMRegressor(**params,
                              cat_feature = cat_list)
        model.fit(tr_X, tr_y)
    
    val_pred = model.predict(val_X)
    val_pred = val_pred.tolist()
    val_RMSE = rmse(val_y, val_pred)
    return(val_RMSE)

    

if __name__ == "__main__":


    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    ############### WANDB OPTION
    arg('--project', type=str, default='book-rating-prediction')
    arg('--entity', type=str, default='recsys01')

    ############### BASIC OPTION
    arg('--data_path', type=str, default='../data/', help='Data path를 설정할 수 있습니다.')
    arg('--data_shuffle', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--test_size', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--seed', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('--model', type=str, default='catboost', choices=['catboost','lgbm'],
                                help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='학습에 사용할 Device를 조정할 수 있습니다.')
    arg('--name', type=str, default=f'work-{get_timestamp()}')
    arg('--saved_model_path', type=str, default='./saved_models', help='Saved Model path를 설정할 수 있습니다.')
    arg('--use_best_model', type=bool, default=True, help='검증 성능이 가장 좋은 모델 사용여부를 설정할 수 있습니다.')
    

    ############### TRAINING OPTION
    arg('--cv', type=str, default= 'Hold_out', choices=['Hold_out', 'K_fold'], help='교차검증 방식을 변경할 수 있습니다.')
    arg('--loss_fn', type=str, default='RMSE', choices=['MSE', 'RMSE'], help='손실 함수를 변경할 수 있습니다.')
    arg('--optimizer', type=str, default='ADAM', choices=['SGD', 'ADAM'], help='최적화 함수를 변경할 수 있습니다.')
    arg('--weight_decay', type=float, default=1e-6, help='Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.')
    arg('--process_cat', type=str, default='basic', choices=['basic', 'high'], help='books 데이터의 카테고리를 선택할 수 있습니다.')
    arg('--process_age', type=str, default='global_mean', choices=['global_mean', 'zero_cat','stratified', 'knn', 'rand_norm'], help='데이터의 결측치를 처리할 방법을 선택할 수 있습니다.')
    arg('--process_loc', type=str, nargs='+', default=['city', 'state', 'country'], choices=['none', 'city', 'state', 'country'], help='usesr의 location을 구분할 기준을 선택할 수 있습니다. none을 선택하면 location은 drop됩니다.')
    
    ############### OPTUNA OPTION
    arg('--n_trials', type = int, default = 20, help = "Search 횟수를 설정할 수 있습니다.")

    args = parser.parse_args()

    ######################## Setting for saving
    setting = Setting(args)

    log_path = setting.get_log_path(args)
    setting.make_dir(log_path)
    filename = setting.get_submit_filename(args)


    ####################### Wandb logging
    wandb_kwargs = {"project": "Tree-based models",
                    "entity" : 'recsys01',
                    'name' : args.name,
                    "reinit": True}
    
    wandbc = WeightsAndBiasesCallback(metric_name="RMSE", wandb_kwargs=wandb_kwargs)
    
    optuna_cbrm = optuna.create_study(direction='minimize', sampler = TPESampler())
    optuna_cbrm.optimize(CB_optuna, n_trials = args.n_trials , callbacks =[wandbc])    

    ####################### data load for prediction
    train, test = catboost_Data(args)


    X_train, y_train = train, train['rating']
    X_test, y_test = test.drop(['rating','book_author'], axis=1), test['rating']


    tr_X, val_X, tr_y, val_y = train_test_split(X_train,
                                                y_train,
                                                test_size = args.test_size,
                                                random_state= args.seed,
                                                shuffle=True)
    
    tr_X = tr_X.drop(['rating'], axis = 1)
    val_result = val_X[['user_id','rating']]
    val_X = val_X.drop([ 'rating'], axis = 1)
    cat_list = [x for x in tr_X.columns.tolist()]
    
    X_train = train.drop(['book_author','rating'], axis = 1)


    f = "best_{}".format
    for param_name, param_value in optuna_cbrm.best_trial.params.items():
        wandb.run.summary[f(param_name)] = param_value
        wandb.run.summary["best val_RMSE"] = optuna_cbrm.best_trial.value
    
    wandb.run.tags = [args.model]
    
    wandb.log({"param_importance_chart" : plot_param_importances(optuna_cbrm) ,
               "param_optimization_history" : plot_optimization_history(optuna_cbrm)})

    ####################### Final train&prediction using best model
    print(f"=============best trial value : {optuna_cbrm.best_trial.value}=============")
    print(f"=============best trial parameter : {optuna_cbrm.best_trial.params}=============")
    


    ####################### Train& predict using best params
    
    best_params = optuna_cbrm.best_trial.params
    if args.model == "catboost":
        best_model = CatBoostRegressor(**best_params,
                                       task_type = "GPU",
                                       cat_features = cat_list,
                                       random_seed= args.seed,
                                       bootstrap_type='Poisson',
                                       verbose = 100)
        # valid prediction
        best_model.fit(X_train, y_train)
        val_result['pred'] = best_model.predict(val_X)

        best_model.fit(X_train, y_train)
        predicts = best_model.predict(X_test)


    elif args.model == 'lgbm':
        tr_X = tr_X.astype('category')
        best_model = LGBMRegressor(**best_params,
                                   cat_feature = cat_list)
        best_model.fit(tr_X, tr_y)

        val_X = val_X.astype('category')
        val_result['pred'] = best_model.predict(val_X)

        X_train = X_train.astype('category')
        X_test = X_test.astype('category')

        best_model.fit(X_train, y_train)
        predicts = best_model.predict(X_test)

    
    

    os.makedirs(args.saved_model_path, exist_ok=True)
    if args.model == 'catboost':
        saved_model_path = f"{args.saved_model_path}/{setting.save_time}_{args.model}_model.cbm"
        best_model.save_model(saved_model_path)
        wandb.save(saved_model_path)
    else:
        pass
        
    
    wandb.save(log_path+f'{args.model}_params.json')
    
    ######################## SAVE PREDICT
    print(f'--------------- PREDICTING {args.model} ---------------')
    submission = pd.read_csv(args.data_path + 'sample_submission.csv')
    submission['rating'] = predicts
    
    print(f'--------------- SAVE {filename} ---------------')
    submission.to_csv(filename, index=False)
    wandb.save(filename)

    val_result.to_csv(filename.replace('.csv', '_valid.csv'), index=False)
    wandb.save(filename.replace('.csv', '_valid.csv'))