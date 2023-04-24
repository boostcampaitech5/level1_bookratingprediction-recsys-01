import argparse
import wandb
import os
import dotenv
import pandas as pd
from src.utils import Logger, Setting, get_timestamp
from src.data import TreeBase_data
from src.TreeBased import grid_search, Valid, OOF, Test
import warnings

def main(args):
    Setting.seed_everything(args.seed)
    warnings.filterwarnings("ignore")
    
    ######################## Load .env
    dotenv.load_dotenv()

    ######################## WANDB
    run = wandb.init(project=args.project, entity=args.entity, name=args.name)
    run.tags = [args.model]
    run.notes = ','.join(args.process_feat)
    
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
        data = TreeBase_data(args)
    else:
        pass

    ######################## find best hyp-param
    print(f'--------------- {args.tuning} parameter tuning ---------------')
    if args.tuning in ('gridcv'):
        best_model = grid_search(data, log_path, args) #return best model
    else:
        pass
    
    ######################## Valid
    print(f'--------------- {args.model} Valid ---------------')
    valid_res = Valid(data, best_model, args)

    
    ######################## Test
    print(f'--------------- {args.model} Test ---------------')
    predicts = Test(data, best_model)
 
    
    ######################## OOF
    print(f'--------------- {args.model} OOF ---------------')
    OOF_res = OOF(data, best_model, args)

    ######################## SAVE Predict and Valid and OOF
    print(f'--------------- SAVE {args.model} PREDICT ---------------')
    submission = pd.read_csv(args.data_path + 'sample_submission.csv')
    submission_valid = pd.read_csv(args.data_path + 'sample_validation.csv')
    submission_oof = submission
    
    if args.model in ('catboost','lgbm'):
        submission['rating'] = predicts
        submission_valid['rating'] = valid_res
        submission_oof['rating'] = OOF_res
    else:
        pass

    pred_name = setting.get_submit_filename(args)
    val_name = pred_name.replace('.csv', '_valid.csv')
    oof_name = pred_name.replace('.csv', '_oof.csv')
    
    submission.to_csv(pred_name, index=False)
    submission_valid.to_csv(val_name, index=False)
    submission_oof.to_csv(oof_name, index=False)
    
    wandb.save(pred_name)
    wandb.save(val_name)
    wandb.save(oof_name)
      
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
    arg('--process_feat', type=str, nargs='+', default=[], choices=['user_id', 'age', 'isbn', 'category', 'publisher', 'language', 'book_author','year_of_publication', 'location_city', 'location_state', 'location_country'], help='Tree model에서 drop할 feature를 고를 수 있습니다.')

    args = parser.parse_args()
    main(args)