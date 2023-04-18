import time
import argparse
import wandb
import os
import dotenv
import config
import json
import pandas as pd
import numpy as np
from src.utils import Logger, Setting, models_load, get_timestamp
from src.data import catboost_Data
from src.TreeBased import grid_search, train_test, prepare_data

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
    if args.model in ('catboost', 'lgbm'):
        data = catboost_Data(args) # data = [train_data, test_data]
    else:
        pass

    ######################## find best hyp-param
    print(f'--------------- {args.tuning} parameter tuning ---------------')
    if args.tuning in ('gridcv'):
        best_model = grid_search(data, log_path, args) #return best model
    else:
        pass
    
    ######################## train and test
    print(f'--------------- {args.model} PREDICT ---------------')
    if args.tuning in ('gridcv'):
        predicts, predicts_oof, valid_res = train_test(data, best_model, args)
    else:
        pass

    ######################## SAVE PREDICT and Valid
    print(f'--------------- SAVE {args.model} PREDICT ---------------')
    submission = pd.read_csv(args.data_path + 'sample_submission.csv')
    submission_oof = submission
    
    if args.model in ('catboost','lgbm'):
        submission['rating'] = predicts
    else:
        pass

    filename = setting.get_submit_filename(args)
    oof_name = filename.replace('.csv', '_oof.csv')
    val_name = filename.replace('.csv', '_valid.csv')
    
    submission.to_csv(filename, index=False)
    submission_oof.to_csv(oof_name, index=False)
    valid_res.to_csv(val_name, index=False)
    
    wandb.save(filename)
    wandb.save(oof_name)
    wandb.save(val_name)
    
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
    arg('--model', type=str, choices=['catboost', 'lgbm'], help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--device', type=str, default='GPU', choices=['GPU', 'CPU'], help='학습에 사용할 Device를 조정할 수 있습니다.')
    arg('--data_shuffle', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--test_size', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    
    ############### TRAINING OPTION
    arg('--tuning', type=str, default='gridcv', choices=['gridcv'], help='하이퍼 파라미터 튜닝 방식을 정할 수 있습니다.')
    arg('--n_fold', type=int, default= 5,  help= '교차검증 방식을 변경할 수 있습니다.')
    arg('--process_cat', type=str, default='basic', choices=['basic', 'high'], help='books 데이터의 카테고리를 선택할 수 있습니다.')
    arg('--process_age', type=str, default='global_mean', choices=['global_mean', 'zero_cat','stratified', 'rand_norm'], help='데이터의 결측치를 처리할 방법을 선택할 수 있습니다.')
    arg('--process_loc', type=str, nargs='+', default=['city', 'state', 'country'], choices=['none', 'city', 'state', 'country'], help='usesr의 location을 구분할 기준을 선택할 수 있습니다. none을 선택하면 location은 drop됩니다.')
    

    args = parser.parse_args()
    main(args)