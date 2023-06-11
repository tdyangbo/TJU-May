from catboost import CatBoostRegressor, Pool
import xgboost as xgb
from params import xgb_params, cb_params, columns_c

def May_cb(train=False):
    if train:
        model = CatBoostRegressor(**cb_params)
    else:
        model = CatBoostRegressor()
    return model

def May_xgb(train=False):
    if train:
        model = xgb.XGBRegressor(**xgb_params)
    else:
        model = xgb.XGBRegressor()
    return model