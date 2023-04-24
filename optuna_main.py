import time
import argparse
import wandb
import os
import dotenv
import config
import optuna
import pandas as pd
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.visualization import plot_param_importances, plot_optimization_history
from sklearn.model_selection import train_test_split
from src.utils import Setting, get_timestamp, rmse
from src.TreeBased import Valid, OOF, Test, get_params, get_wandb_args
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor 
from src.data import TreeBase_data
import warnings
warnings.filterwarnings("ignore")

def CB_optuna(trial:Trial):
    params = get_params(args, trial)

    ######################## Load Dataset
    data = TreeBase_data(args)
    cat_list = data[-1]
    tr_X, val_X, tr_y, val_y = train_test_split(data[0],
                                                    data[1],
                                                    test_size = 0.2,
                                                    random_state= args.seed,
                                                    shuffle=True
                                                        )
    
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
                              cat_feature = cat_list,
                              early_stopping_rounds=10)
        model.fit(tr_X, tr_y, eval_set = (val_X, val_y),eval_metric = 'rmse', verbose = 500)
    
    val_pred = model.predict(val_X)
    val_pred = val_pred.tolist()
    val_RMSE = rmse(val_y, val_pred)
    return(val_RMSE)

    

if __name__ == "__main__":


    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

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
    arg('--n_fold', type=int, default= 5,  help= '교차검증 방식을 변경할 수 있습니다.')
    arg('--cv', type=str, default= 'Hold_out', choices=['Hold_out', 'K_fold'], help='교차검증 방식을 변경할 수 있습니다.')
    arg('--process_cat', type=str, default='basic', choices=['basic', 'high'], help='books 데이터의 카테고리를 선택할 수 있습니다.')
    arg('--process_age', type=str, default='global_mean', choices=['global_mean', 'zero_cat','stratified', 'knn', 'rand_norm'], help='데이터의 결측치를 처리할 방법을 선택할 수 있습니다.')
    arg('--process_feat', type=str, nargs='+', default=[], choices=['user_id', 'age', 'isbn', 'category', 'publisher', 'language', 'book_author','year_of_publication', 'location_city', 'location_state', 'location_country'], help='Tree model에서 drop할 feature를 고를 수 있습니다.')
    
    ############### OPTUNA OPTION
    arg('--n_trials', type = int, default = 20, help = "Search 횟수를 설정할 수 있습니다.")

    args = parser.parse_args()

    ######################## Setting for saving
    setting = Setting(args)

    log_path = setting.get_log_path(args)
    setting.make_dir(log_path)
    filename = setting.get_submit_filename(args)


    ####################### Wandb logging
    wandb_kwargs = get_wandb_args(args)
    
    wandbc = WeightsAndBiasesCallback(metric_name="RMSE", wandb_kwargs=wandb_kwargs)
    
    optuna_cbrm = optuna.create_study(direction='minimize', sampler = TPESampler())
    optuna_cbrm.optimize(CB_optuna, n_trials = args.n_trials , callbacks =[wandbc])    

    ####################### data load for prediction
    data = TreeBase_data(args) 

    cat_list = data[-1]
    tr_X, val_X, tr_y, val_y = train_test_split(data[0],
                                                data[1],
                                                test_size = args.test_size,
                                                random_state= args.seed,
                                                shuffle=True)

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
                                       bootstrap_type='MVS',
                                       verbose = 100)
        best_model.fit(tr_X, tr_y)
        
    elif args.model == 'lgbm':
        best_model = LGBMRegressor(**best_params,cat_feature = cat_list)
        best_model.fit(tr_X, tr_y, eval_set = (val_X, val_y),eval_metric = 'rmse', verbose = 500, early_stopping_rounds=10)
    
    

    valid_pred = Valid(data, best_model, args)
    oof_pred = OOF(data, best_model, args)
    predicts = Test(data, best_model)
    

    os.makedirs(args.saved_model_path, exist_ok=True)
    if args.model == 'catboost':
        saved_model_path = f"{args.saved_model_path}/{setting.save_time}_{args.model}_model.cbm"
        best_model.save_model(saved_model_path)
        wandb.save(saved_model_path)
    else:
        pass
        
    
    wandb.save(log_path+f'{args.model}_params.json')
    
    ######################## SAVE RESULTS

    print(f'--------------- SAVING RESULTS  ---------------')
    submission = pd.read_csv(args.data_path + 'sample_submission.csv')
    submission['rating'] = predicts
    submission.to_csv(filename, index=False)
    wandb.save(filename)
    
    
    oof_submission = pd.read_csv(args.data_path + 'sample_submission.csv')
    oof_submission['rating'] = oof_pred
    oof_submission.to_csv(filename.replace('.csv', '_oof.csv'), index=False)
    wandb.save(filename.replace('.csv', '_oof.csv'))
    

    val_result = pd.read_csv(args.data_path + 'sample_validation.csv')
    val_result['pred'] = valid_pred
    val_result.to_csv(filename.replace('.csv', '_valid.csv'), index=False)
    wandb.save(filename.replace('.csv', '_valid.csv'))
    
    
