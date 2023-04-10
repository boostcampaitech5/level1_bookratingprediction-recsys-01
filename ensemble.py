import pandas as pd
import numpy as np

from src.ensembles.ensembles import Ensemble
import argparse

def main(args):
    file_list = sum(args.ensemble_files, [])
    
    if len(file_list) < 2:
        raise ValueError("Ensemble할 Model을 적어도 2개 이상 입력해 주세요.")
    
    en = Ensemble(filenames = file_list,filepath=args.result_path)

    if args.ensemble_strategy == 'weighted':
        if args.ensemble_weight: 
            strategy_title = 'sw-'+'-'.join(map(str,*args.ensemble_weight)) #simple weighted
            result = en.simple_weighted(*args.ensemble_weight)
        else:
            strategy_title = 'aw' #average weighted
            result = en.average_weighted()
    elif args.ensemble_strategy == 'mixed':
        strategy_title = args.ensemble_strategy.lower() #mixed
        result = en.mixed()
    else:
        pass
    en.output_frame['rating'] = result
    output = en.output_frame.copy()
    files_title = '-'.join(file_list)

    output.to_csv(f'{args.result_path}{strategy_title}-{files_title}.csv',index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument
    '''
    [실행 방법]
    ```
    python ensemble.py [인자]
    ```
    
    [인자 설명]
    > 스크립트 실행 시, 
    > 인자가 필수인 경우 required
    > 필수가 아닌 경우, optional 로 명시하였습니다.
    
    --ensemble_files ensemble_files [ensemble_files ...]
    required: 앙상블할 submit 파일명을 쉼표(,)로 구분하여 모두 입력해 주세요. 
    이 때, 경로(submit)와 확장자(.csv)는 입력하지 않습니다.

    --ensemble_strategy {weighted,mixed}
    optional: 앙상블 전략을 선택해 주세요.
    [mixed, weighted] 중 선택 가능합니다.
    (default="weighted")

    --ensemble_weight ensemble_weight [ensemble_weight ...]
    optional: weighted 앙상블 전략에서만 사용되는 인자입니다.
    전달받은 결과값의 가중치를 조정할 수 있습니다.
    가중치를 쉼표(,)로 구분하여 모두 입력해 주세요.
    이 때, 합산 1이 되지 않는 경우 작동하지 않습니다.

    --result_path result_path
    optional: 앙상블할 파일이 존재하는 경로를 전달합니다. 
    기본적으로 베이스라인의 결과가 떨어지는 공간인 submit으로 연결됩니다.
    앙상블된 최종 결과물도 해당 경로 안에 떨어집니다.
    (default:"./submit/")

    [결과물]
    result_path 안에 앙상블된 최종 결과물이 저장됩니다.
    {strategy_title}-{files_title}.csv

    파일명은 ensemble_files과 ensemble_strategy가 모두 명시되어 있습니다.
    ensemble_strategy의 경우, 아래와 같이 작성됩니다.
    > simple weighted : sw + 각 파일에 적용된 가중치
    > average weighted : aw
    > mixed : mixed
    '''

    arg("--ensemble_files", nargs='+',required=True,
        type=lambda s: [item for item in s.split(',')],
        help='required: 앙상블할 submit 파일명을 쉼표(,)로 구분하여 모두 입력해 주세요. 이 때, .csv와 같은 확장자는 입력하지 않습니다.')
    arg('--ensemble_strategy', type=str, default='weighted',
        choices=['weighted','mixed'],
        help='optional: [mixed, weighted] 중 앙상블 전략을 선택해 주세요. (default="weighted")')
    arg('--ensemble_weight', nargs='+',default=None,
        type=lambda s: [float(item) for item in s.split(',')],
        help='optional: weighted 앙상블 전략에서 각 결과값의 가중치를 조정할 수 있습니다.')
    arg('--result_path',type=str, default='./submit/',
        help='optional: 앙상블할 파일이 존재하는 경로를 전달합니다. (default:"./submit/")')
    args = parser.parse_args()
    main(args)