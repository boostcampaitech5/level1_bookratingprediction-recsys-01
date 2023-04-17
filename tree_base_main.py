import time
import argparse
import wandb
import pandas as pd
import os
import dotenv
import config
from functools import partial
from src.utils import Logger, Setting, get_timestamp
from src.data import catboost_Data #, naiveB_Data
from src.TreeBased import grid_search, optuna_search, train_test

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
        
    elif args.model in ('naiveB'):
        data = naiveB_Data(args)
        
    else:
        pass

    ######################## find best hyp-param
    print(f'--------------- {args.} parameter tuning ---------------')
    if args.tuning in ('gridcv'):
        best_model = grid_search(data, args) #return best model
        
    elif args.tuning in ('optuna'):
        best_params = optuna_search(data, args)
        
    else:
        pass
    
    ######################## train and test
    print(f'--------------- {args.model} PREDICT ---------------')
    predicts = train_test(args, best_model, data)
    



    ######################## Model & loss_fn & optimizer
    print(f'--------------- INIT {args.model} ---------------')
    model = models_load(args,data)
    loss_fn = loss_fn_load(args)
    optimizer = optimizer_load(args,model)


    ######################## TRAIN
    print(f'--------------- {args.model} TRAINING ---------------')
    if args.cross_validation:
        model = cv_train(args, model, data, loss_fn, optimizer, logger, setting)
    else:
        model = train(args, model, data, loss_fn, optimizer, logger, setting)


    ######################## INFERENCE
    print(f'--------------- {args.model} PREDICT ---------------')
    predicts = test(args, model, data, setting)
    

    ######################## SAVE PREDICT
    print(f'--------------- SAVE {args.model} PREDICT ---------------')
    submission = pd.read_csv(args.data_path + 'sample_submission.csv')
    if args.model in ('FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN', 'DeepCoNN_CNN', 'FFDCN'):
        submission['rating'] = predicts
    else:
        pass

    filename = setting.get_submit_filename(args)
    submission.to_csv(filename, index=False)
    wandb.save(filename)
    

    ######################## INFERENCE & SAVE VALID
    print(f'--------------- INFERENCE & SAVE {args.model} VALID ---------------')
    result = infer(args, model, data, setting)
    valid_filename = filename.replace('.csv', '_valid.csv')
    result.to_csv(valid_filename, index=False)
    wandb.save(valid_filename)


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
    arg('--model', type=str, choices=['catboost', 'naiveB'], help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--device', type=str, default='GPU', choices=['GPU', 'CPU'], help='학습에 사용할 Device를 조정할 수 있습니다.')
    arg('--data_shuffle', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    

    

    ############### TRAINING OPTION
    arg('--tuning', type=str, default=['gridcv', 'optuna'], help='하이퍼 파라미터 튜닝 방식을 정할 수 있습니다.')
    arg('--n_fold', type=int, default= 5,  help= '교차검증 방식을 변경할 수 있습니다.')
    arg('--process_cat', type=str, default='basic', choices=['basic', 'high'], help='books 데이터의 카테고리를 선택할 수 있습니다.')
    arg('--process_age', type=str, default='global_mean', choices=['global_mean', 'zero_cat','stratified', 'rand_norm'], help='데이터의 결측치를 처리할 방법을 선택할 수 있습니다.')
    arg('--process_loc', type=str, nargs='+', default=['city', 'state', 'country'], choices=['none', 'city', 'state', 'country'], help='usesr의 location을 구분할 기준을 선택할 수 있습니다. none을 선택하면 location은 drop됩니다.')
    

    args = parser.parse_args()
    CatBoost_cv(args)