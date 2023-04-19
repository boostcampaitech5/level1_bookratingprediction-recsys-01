import wandb
import config
import json
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from src.data import catboost_Data
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

def prepare_data(data):
    train_data, test_data = data[0], data[1]
    X_train_data, y_train_data = train_data, train_data['rating']
    X_test_data, y_test_data = test_data.drop(['rating'], axis=1), test_data['rating']

    return X_train_data, y_train_data, X_test_data, y_test_data

def train_test (data, model, args):
    
    ######################## setting data
    X_train_data, y_train_data, X_test_data, y_test_data = prepare_data(data)
    
    ######################## valid
    train_x, vali_x, train_y, valid_y = train_test_split(X_train_data,
                                                        y_train_data,
                                                        test_size=args.test_size,
                                                        random_state=args.seed,
                                                        shuffle=args.data_shuffle)
    val_result = vali_x[['user_id', 'isbn', 'rating']]
    vali_x = vali_x.drop(['rating'], axis=1)
    train_x = train_x.drop(['rating'], axis=1)
    X_train_data = X_train_data.drop(['rating'], axis=1)

    cat_list = [x for x in X_train_data.columns.tolist()]

    if args.model == 'lgbm':
        vali_x = vali_x.astype('category')
        train_x[cat_list] = train_x[cat_list].astype('category')
    
    model.fit(train_x, train_y)
    valid_pred = model.predict(vali_x)
    
    val_result['pred'] = valid_pred
    
    score = []
    
    if args.model == 'lgbm':
        X_test_data = X_test_data.astype('category')
        X_train_data[cat_list] = X_train_data[cat_list].astype('category')
    
    cv = StratifiedKFold(n_splits= args.n_fold, shuffle=args.data_shuffle, random_state=args.seed)
    for train_idx, test_idx in cv.split(X_train_data, y_train_data):
        X_train, y_train = X_train_data.iloc[train_idx], y_train_data.iloc[train_idx]
        model.fit(X_train_data, y_train_data)
        score.append(model.predict(X_test_data).tolist())
        
    y_hat_oof = list(np.sum(score, axis= 0) / args.n_fold)
                
    ######################## testing
    model.fit(X_train_data, y_train_data)
    y_hat = model.predict(X_test_data)
    return y_hat.tolist(), y_hat_oof, val_result