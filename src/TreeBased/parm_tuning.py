import config
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from catboost import CatBoostRegressor, Pool


def grid_search(data, args):
    ######################## setting data
    train_data, test_data = data[0], data[1]
    X_train_data, y_train_data = train_data.drop(['user_id', 'isbn', 'rating','book_author'], axis=1), train_data['rating']
    X_test_data, y_test_data = test_data.drop(['user_id', 'isbn', 'rating','book_author'], axis=1), test_data['rating']
    
    cat_list = [x for x in X_train_cat.columns.tolist()]
    
    ######################## Model select
    if args.model in ('catboost'):
        model = CatBoostRegressor(cat_features=cat_list, train_dir = log_path+'catboost_info')
    elif args.model in ('naiveB')
        # model=~
    #config 파일 불러오기 추가
    
    cv = StratifiedKFold(n_splits= args.n_fold, shuffle=args.data_shuffle, random_state=args.seed)
    grid = GridSearchCV(estimator=model, cv = cv, param_grid=params)
    grid.fit(X_train_data, y_train_data)

    wandb.log({'best_valid_performance': cb_grid.best_score_})
    
    print(f"====={args.n_fold}-Fold CV result - valid RMSE : {cb_grid.best_score_}=====")
    print(f"====={args.n_fold}-Fold CV result - best parameters : {cb_grid.best_params_}=====")
    
    f = "best_{}".format
    saving_param = catboost_cl.get_params()
    for param_name, param_value in cb_grid.best_estimator_:
        wandb.run.summary[f(param_name)] = param_value
    
    return cb_grid.best_estimator_



def optuna_search():
    