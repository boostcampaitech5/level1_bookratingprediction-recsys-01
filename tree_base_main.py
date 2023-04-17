import time
import argparse
import wandb
import pandas as pd
import os
import dotenv
import config
import optuna
from optuna import Trial
from functools import partial
from src.utils import Logger, Setting, get_timestamp
from src.data import catboost_Data #, naiveB_Data
from src.TreeBased import grid_search, optuna_search, train_test, prepare_data
from optuna.samplers import TPESampler
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.visualization import plot_param_importances, plot_optimization_history


def optuna_obj(trial:Trial):
    if args.model in ('catboost'):
        data = catboost_Data(args)
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



def main(args):
    Setting.seed_everything(args.seed)
    
    ######################## Load .env
    dotenv.load_dotenv()

    ######################## WANDB
    run = wandb.init(project=args.project, entity=args.entity, name=args.name)
    run.tags = [args.model]
    
    ####################### Setting for 
    setting = Setting(args)

    log_path = setting.get_log_path(args)
    setting.make_dir(log_path)

    logger = Logger(args, log_path)
    logger.save_args()
    
    ####################### WANDB
    WANDB_API_KEY = os.environ.get('WANDB_API_KEY')
    wandb.login(key=WANDB_API_KEY)

    ######################## DATA LOAD
    print(f'--------------- {args.model} Load Data ---------------')
    if args.model in ('catboost'):
        data = catboost_Data(args) # data = [train_data, test_data]
        
        '''elif args.model in ('naiveB'):
            print('haha')'''

    else:
        pass

    ######################## find best hyp-param
    print(f'--------------- {args.tuning} parameter tuning ---------------')
    if args.tuning in ('gridcv'):
        best_model = grid_search(data, log_path, args) #return best model
        
    elif args.tuning in ('optuna'):
        #best_params =  optuna_search(data, args)

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

        best_params = optuna_cbrm.best_trial.params

        cat_list = [x for x in X_train_cat.columns.tolist()]
        X_train_cat, y_train_cat = data[0].drop(['user_id', 'isbn', 'rating','book_author'], axis=1), train_cat['rating']
        X_test_cat, y_test_cat = data[1].drop(['user_id', 'isbn', 'rating','book_author'], axis=1), test_cat['rating']

        best_model = CatBoostRegressor(**best_params,
                                   #task_type = "GPU",
                                   cat_features = cat_list,
                                   random_seed= args.seed,
                                   #bootstrap_type='Poisson',
                                   verbose = 100)
        
        best_model.fit(X_train_cat, y_train_cat)
        
    else:
        pass
    
    ######################## train and test
    print(f'--------------- {args.model} PREDICT ---------------')
    if args.tuning in ('gridcv'):
        predicts = train_test(data, best_model, args)

    elif args.tuning in ('optuna'):
        predicts = best_model.predict(X_test_cat)

    ######################## SAVE PREDICT
    print(f'--------------- SAVE {args.model} PREDICT ---------------')
    submission = pd.read_csv(args.data_path + 'sample_submission.csv')
    if args.model in ('catboost','model2'):
        submission['rating'] = predicts
    else:
        pass

    filename = setting.get_submit_filename(args)
    submission.to_csv(filename, index=False)
    wandb.save(filename)


    ######################## WANDB & Logger FINISH
    logger.close()
    wandb.finish()

if __name__ == "__main__":


    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument
    
    ############### WANDB OPTION
    arg('--project', type=str, default='Tree-based models')
    arg('--entity', type=str, default='recsys01')
    arg('--name', type=str, default=f'work-{get_timestamp()}')

    ############### BASIC OPTION
    arg('--data_path', type=str, default='../data/', help='Data path를 설정할 수 있습니다.')
    arg('--saved_model_path', type=str, default='./saved_models', help='Saved Model path를 설정할 수 있습니다.')
    arg('--seed', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('--model', type=str, choices=['catboost', 'model2'], help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--device', type=str, default='GPU', choices=['GPU', 'CPU'], help='학습에 사용할 Device를 조정할 수 있습니다.')
    arg('--data_shuffle', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    
    ############### TRAINING OPTION
    arg('--tuning', type=str, default=['gridcv', 'optuna'], help='하이퍼 파라미터 튜닝 방식을 정할 수 있습니다.')
    arg('--n_fold', type=int, default= 5,  help= '교차검증 방식을 변경할 수 있습니다.')
    arg('--process_cat', type=str, default='basic', choices=['basic', 'high'], help='books 데이터의 카테고리를 선택할 수 있습니다.')
    arg('--process_age', type=str, default='global_mean', choices=['global_mean', 'zero_cat','stratified', 'rand_norm'], help='데이터의 결측치를 처리할 방법을 선택할 수 있습니다.')
    arg('--process_loc', type=str, nargs='+', default=['city', 'state', 'country'], choices=['none', 'city', 'state', 'country'], help='usesr의 location을 구분할 기준을 선택할 수 있습니다. none을 선택하면 location은 drop됩니다.')
    
    ############### OPTUNA OPTION
    arg('--n_trials', type = int, default = 20, help = "Search 횟수를 설정할 수 있습니다.")


    args = parser.parse_args()
    main(args)