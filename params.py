xgb_params = {
    'eval_metric':'rmse',
    'booster': 'gbtree',
    'objective': 'reg:gamma',
    'gamma': 0.1,
    'max_depth': 9,
    'lambda': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'slient': 0,
    'learning_rate': 0.03,
    'seed': 2023,
    # 'nthread': 4,
}

cb_params = {
    'objective': 'RMSE',
    'eval_metric': 'MAE',
}

cb_params_1 = {
    'objective': 'RMSE',
    'eval_metric': 'MAE',
    'learning_rate': 0.03,
    'l2_leaf_reg': 3,
    'max_ctr_complexity': 1,
    'depth': 8,
    'leaf_estimation_method': 'Gradient',
    'use_best_model': True,
    'iterations': 100000,
    'early_stopping_rounds': 5000,
    'verbose': 500
}

cb_params_2 = {
    'objective': 'RMSE',
    'eval_metric': 'MAE',
    'learning_rate': 0.05,
    'l2_leaf_reg': 3,
    'max_ctr_complexity': 1,
    'depth': 6,
    'leaf_estimation_method': 'Gradient',
    'use_best_model': True,
    'early_stopping_rounds': 1000,
    'verbose': 100
}

columns_c = ['Uid', 'Category', 'Subcategory', 'Concept', 'Mediatype', 'hour', 'day', 'weekday', 'week_hour',
             'year_weekday', 'Geoaccuracy', 'ispro', 'Ispublic', 'img_model']