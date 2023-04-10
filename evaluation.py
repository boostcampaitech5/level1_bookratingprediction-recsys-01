import os
import json

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(y, pred):
    y, pred = y.values, pred.values
    mse = np.mean((y-pred)**2)
    return mse**0.5

def evaluation(gt_path, pred_path):
        """
        Args:
          gt_path (string) : root directory of ground truth file (gt.csv or private.csv)
          pred_path (string) : root directory of prediction file (output.csv)
        """
        
        gt = pd.read_csv(os.path.join(gt_path))
        pred = pd.read_csv(os.path.join(pred_path))
        pred = pred.reset_index()
        pred = pred[pred['index'].isin(gt['index'])]
        
        assert len(gt) == len(pred), f'Assertion Failed {len(gt)} != {len(pred)}'

        """
        TODO : metric 코드 작성
        missing/inf value exception logic 필요?
        """
        
        value = rmse(gt['rating'], pred['rating'])
        
        results = {}  # a dictionary where the keys are the metric names
        results['RMSE'] = {
                'value': f'{value:.04f}',   # metric score
                'rank': True,               # True if used for ranking, False otherwise
                'decs': False,                # True for descending order, False otherwise
            }
        
        # results['METRIC_NAME_2'] = {
        #     'value': f'{value2:.04f}',
        #     'rank': False,
        #     'decs': True,
        # }

        return results