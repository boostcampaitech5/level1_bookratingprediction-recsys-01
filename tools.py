import pandas as pd
from typing import Dict
from skearn.metrics import mean_squared_error


def get_valid_result(name: str) -> Dict:
    '''
    [description]
    실험의 vaildation 데이터에 대한 예측결과를 딕셔너리로 반환하는 함수입니다.
    (주의) root 경로에 있는 노트북에서 실행해야 정상적으로 동작합니다.

    [arguments]
    name : 실험명 ex) 'work-230413_030817_NCF'
    '''
    path = f'./level1_bookratingprediction-recsys-01/submit/{name}_valid.csv'
    data = pd.read_csv(path)
    result = dict()
    result['data'] = data
    result['RMSE'] = mean_squared_error(data['rating'], data['pred']) ** 0.5
    print(f'valid result for experiment: {name} is loaded')
    return result
