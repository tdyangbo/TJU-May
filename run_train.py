'''
The file is used to train a new model with the given features
'''

from dataloader import load_all_feat, split_data
from catboost import Pool
from params import cb_params, columns_c
from sklearn.model_selection import KFold
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import save_submission
from May_model import *
import json

def train_model(train_feat,train_label,test_data):
    '''
    :param train_feat: all training feature from the loading
    :param train_label: corresponding label of all training data
    :param test_data: the testing feature
    :return: test_proba, test_label
    '''

    valid_src = 0
    valid_mse = 0
    valid_mae = 0
    valid_ans = []
    test_proba = []

    kfold = KFold(n_splits=3, shuffle=True, random_state=2023)
    k = 0
    print('===================training=====================')

    for train_idx, valid_idx in kfold.split(train_feat, train_label):
        # kfold split data
        fold_train_x, fold_train_y = train_feat.loc[train_idx], train_label['label'].loc[train_idx]
        fold_valid_x, fold_valid_y = train_feat.loc[valid_idx], train_label['label'].loc[valid_idx]

        # loading the model
        #
        # model = May_xgb()
        # model.fit(fold_train_y, fold_train_y)
        #
        model = May_cb(train=True)
        train_data = Pool(data=fold_train_x, label=fold_train_y, cat_features=columns_c)
        valid_data = Pool(data=fold_valid_x, label=fold_valid_y, cat_features=columns_c)
        model.fit(train_data, eval_set=valid_data)

        # predict the valid data
        valid_pred = model.predict(valid_data)
        mse = mean_squared_error(fold_valid_y, valid_pred)
        mae = mean_absolute_error(fold_valid_y, valid_pred)
        src = stats.spearmanr(fold_valid_y, valid_pred)[0]
        print('===================validation=====================')
        print("MSE: %.4f, MAE: %.4f, SRC: %.4f" % (mse, mae, src))

        # save each fold model
        model.save_model('checkpoint/check/May_model_' + str(k) + '.pkl')

        # optimize and save the best model/result
        if src > valid_src:
            valid_src = src
            valid_mse = mse
            valid_mae = mae
            valid_ans.append([valid_mse, valid_mae, valid_src])
            test_pred = model.predict(test_data)
            test_proba.append(test_pred)
            #
            model.save_model('checkpoint/check/May_best_model.pkl')
        k += 1

    # # mean
    # valid_ans = np.mean(valid_ans, axis=0)
    # print("valid: MSE: %.4f, MAE: %.4f, SRC: %.4f" % (valid_ans[0], valid_ans[1], valid_ans[2]))
    return test_proba, test_label


if __name__ == '__main__':
    data_path = 'data/feature/'

    # loading the pretrained feature/data
    all_feat = load_all_feat(data_path)
    train_feat, train_label, test_feat, test_label = split_data(all_feat)
    test_data = Pool(data=test_feat, label=test_label['label'], cat_features = columns_c)

    # model training
    test_proba, test_label = train_model(train_feat,train_label,test_data)

    # save the submission result file
    save_submission(test_proba, test_label)


