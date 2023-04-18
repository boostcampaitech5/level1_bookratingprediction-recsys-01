import pandas as pd
import numpy as np

from src.ensembles.stacking import Stacking
from src.utils import Setting
import argparse

def main(args):
    Setting.seed_everything(args.seed)
    
    file_list = sum(args.ensemble_files, [])
    
    if len(file_list) < 2:
        raise ValueError("Ensemble할 Model을 적어도 2개 이상 입력해 주세요.")
    
    stacking = Stacking(filenames = file_list,filepath=args.result_path,seed=args.seed,test_size=args.test_size)
    
    ################# Train
    if args.cross_validation:
        stacking.cv_train(stacking.X_train, stacking.y_train)
    else:
        stacking.train(stacking.X_train, stacking.y_train)
    
    
    ################# TEST
    print('----- total test -----')
    stacking.test(stacking.X_test, stacking.y_test)
    
    
    ################# SAVE
    weights = stacking.get_weights()
    result = stacking.infer()
    
    output = stacking.output_frame.copy()
    output['rating'] = result
    
    weight_info = '-'.join([str(w)[:4] for w in weights])
    files_title = '-'.join(file_list)
    
    csv_path = f'{args.result_path}stacked-sw-{weight_info}-{files_title}.csv'
    output.to_csv(csv_path,index=False)
    print(f'========= new output saved : {csv_path} =========')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument
    '''
    [실행 방법]
    ```
    python stacking.py [인자]
    ```
    
    [인자 설명]
    > 스크립트 실행 시, 
    > 인자가 필수인 경우 required
    > 필수가 아닌 경우, optional 로 명시하였습니다.
    
    --ensemble_files ensemble_files [ensemble_files ...]
    required: 앙상블할 submit 파일명을 쉼표(,)로 구분하여 모두 입력해 주세요. 
    이 때, 경로(submit)와 확장자(.csv)는 입력하지 않습니다.

    --result_path result_path
    optional: 앙상블할 파일이 존재하는 경로를 전달합니다. 
    기본적으로 베이스라인의 결과가 떨어지는 공간인 submit으로 연결됩니다.
    앙상블된 최종 결과물도 해당 경로 안에 떨어집니다.
    (default:"./submit/")

    [결과물]
    result_path 안에 앙상블된 최종 결과물이 저장됩니다.
    stacked-sw-{weight_info}-{files_title}.csv
    '''

    arg('--seed', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('-ts', '--test_size', type=float, default=0.5, help='Valid data의 train/test split 비율을 조정할 수 있습니다.')
    arg('-f', "--ensemble_files", nargs='+',required=True,
        type=lambda s: [item for item in s.split(',')],
        help='required: 앙상블할 submit 파일명을 쉼표(,)로 구분하여 모두 입력해 주세요. 이 때, .csv와 같은 확장자는 입력하지 않습니다.')
    arg('-p', '--result_path',type=str, default='./submit/',
        help='optional: 앙상블할 파일이 존재하는 경로를 전달합니다. (default:"./submit/")')
    arg('--sampler', type=str, default=None, choices=[None, 'None', 'weighted'], help='dataloader의 sampler를 변경할 수 있습니다.')
    arg('-cv', '--cross_validation', type=bool, default=False, help='cross validation을 사용할 수 있습니다.')
    args = parser.parse_args()
    main(args)