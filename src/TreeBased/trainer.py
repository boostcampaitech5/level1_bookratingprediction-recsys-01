import wandb
import numpy as np
from src.utils import rmse
from lightgbm import LGBMRegressor 
from catboost import CatBoostRegressor
from src.data import TreeBase_data_split
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV



def Valid (data, model, args):
    ######################## setting data
    X_train_data, y_train_data, _, _, _ = data
    
     ######################## valid
    train_x, valid_x, train_y, valid_y = train_test_split(X_train_data,
                                                        y_train_data,
                                                        test_size=args.test_size,
                                                        random_state=args.seed,
                                                        shuffle=args.data_shuffle)
    
    model.fit(train_x, train_y)
    valid_pred = model.predict(valid_x)
    valid_loss = rmse(valid_y, valid_pred)
    wandb.run.summary['best val_RMSE'] = valid_loss
    
        
    return valid_pred.tolist()
    
def OOF (data, model, args):
    
    ######################## setting data
    X_train_data, y_train_data, X_test_data, y_test_data, _ = data
    
    score = []
    
    cv = StratifiedKFold(n_splits= args.n_fold, shuffle=args.data_shuffle, random_state=args.seed)
    
    for train_idx, test_idx in cv.split(X_train_data, y_train_data):
        X_train, y_train = X_train_data.iloc[train_idx], y_train_data.iloc[train_idx]
        model.fit(X_train_data, y_train_data)
        score.append(model.predict(X_test_data).tolist())
        
    y_hat_oof = list(np.sum(score, axis= 0) / args.n_fold)
    
    return y_hat_oof
    

def Test (data, model):
    
    ######################## setting data
    X_train_data, y_train_data, X_test_data, y_test_data, _ = data
                
    ######################## testing
    model.fit(X_train_data, y_train_data)
    y_hat = model.predict(X_test_data)
    
    return y_hat.tolist()


def model_optuna(args, data, params):
    tr_X, val_X, tr_y, val_y = TreeBase_data_split(data, args)
    cat_list = data[-1]
    if args.model == 'catboost':
        model = CatBoostRegressor(**params,
                                  task_type = "GPU",
                                  cat_features = cat_list,
                                  random_seed= args.seed,
                                  bootstrap_type='MVS',
                                  verbose = 100)
        model.fit(tr_X,tr_y, use_best_model = True, eval_set = (val_X, val_y))
    
    elif args.model == 'lgbm':
        model = LGBMRegressor(**params,
                              cat_feature = cat_list)
        model.fit(tr_X, tr_y, eval_metric = 'rmse', verbose = 500)

    return model