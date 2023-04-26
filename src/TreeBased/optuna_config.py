from optuna import Trial
def get_params(args, trial:Trial):
    if args.model == 'catboost':
        params = {
            'iterations':trial.suggest_int("iterations", 5, 10),
            'learning_rate' : trial.suggest_float('learning_rate',0.001, 0.1),
            'reg_lambda': trial.suggest_float('reg_lambda',50,150),
            'subsample': trial.suggest_float('subsample',0.5,1),
            'random_strength': trial.suggest_float('random_strength',35,55),
            'depth': trial.suggest_int('depth',5, 13),
            #'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',20,50),
            #'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',5,25),
            #'bagging_temperature' :trial.suggest_float('bagging_temperature', 0.01, 100.00)
            }
    
    elif args.model == 'lgbm':
        params = {
            'objective': 'regression',
            'verbosity': -1,
            'metric': 'rmse', 
            'num_iterations' : 10, #trial.suggest_int('num_iterations', 1000, 2000),
            'num_leaves' : 9 , #trial.suggest_int('num_leaves', 3, 9),
            'max_depth': trial.suggest_int('max_depth',4, 9),
            'learning_rate' : trial.suggest_float('learning_rate',0.01, 0.1),
            'n_estimators': trial.suggest_int('n_estimators', 3, 20),
            'min_child_samples': trial.suggest_int('min_child_samples', 70, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.0, 1.0)
            #'reg_alpha' : trial.suggest_float('reg_alpha',  0.0, 1.0),
            #'reg_lambda' : trial.suggest_float('reg_lambda',  0.5, 1.0)
            }
    return params

def get_wandb_args(args):
    wandb_kwargs = {"project": "Tree-based models",
                    "entity" : 'recsys01',
                    "notes" : ','.join(args.process_feat),
                    'name' : args.name,
                    "reinit": True}
    
    return wandb_kwargs