import wandb
import json
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

import warnings

def grid_search(data, log_path, args):
    ######################## setting data
    X_train_data, y_train_data, X_test_data, y_test_data, cat_list = data
    print(cat_list)
    ######################## Model select
    if args.model in ('catboost'):
        model = CatBoostRegressor(cat_features=cat_list, 
                                train_dir = log_path+'catboost_info',
                                task_type = args.device,
                                random_seed= args.seed,
                                bootstrap_type='Poisson'
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
    
    print(f"====={args.n_fold}-Fold CV result - best_score : {grid.best_score_}=====")
    print(f"====={args.n_fold}-Fold CV result - best parameters : {grid.best_params_}=====")
    
    saving_param = grid.best_params_
   ######################## Saving best parm to wandb
    f = "best_{}".format
    for s_parm in saving_param.items():
        wandb.run.summary[f(s_parm[0])] = s_parm[1]
        
    ######################## Saving best parm.json 
    with open(log_path+'best_parm.json', 'w') as f:
        json.dump(saving_param, f) 
    wandb.save(log_path+'best_parm.json')
    
    return grid.best_estimator_
