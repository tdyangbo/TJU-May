'''
The file is used to test the offline metrics of our given models
'''

from dataloader import load_all_feat, split_data
from catboost import Pool
from params import columns_c
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import *
from May_model import *
import json

def test_model(train_feat,train_label,test_data,model,test=False):
    '''
    :param train_feat: all training feature from the loading
    :param train_label: corresponding label of all training data
    :param test_data: the testing feature
    :param model: pretained model
    :param test: default = False
    :return:
    '''
    print('===================validation=====================')
    valid_ans = []
    test_proba = []
    kfold = KFold(n_splits=3, shuffle=True, random_state=2023)
    k = 0
    for train_idx, valid_idx in kfold.split(train_feat, train_label):
        fold_valid_x, fold_valid_y = train_feat.loc[valid_idx], train_label['label'].loc[valid_idx]
        valid_data = Pool(data=fold_valid_x, label=fold_valid_y, cat_features=columns_c)

        model.load_model('checkpoint/May_model_' + str(k) + '.pkl')

        valid_pred = model.predict(valid_data)
        valid_mse = mean_squared_error(fold_valid_y, valid_pred)
        valid_mae = mean_absolute_error(fold_valid_y, valid_pred)
        valid_src = stats.spearmanr(fold_valid_y, valid_pred)[0]

        print('The '+ str(k) +' flod:')
        print("MSE: %.4f, MAE: %.4f, SRC: %.4f" % (valid_mse, valid_mae, valid_src))
        valid_ans.append([valid_mse, valid_mae, valid_src])

        if test:
            print('===================testing=====================')
            test_pred = model.predict(test_data)
            test_proba.append(test_pred)
            # save_checkpoint_result(test_proba,k)

        k += 1


if __name__ == '__main__':
    data_path = 'data/feature/'

    # loading the pretrained feature
    all_feat = load_all_feat(data_path)
    train_feat,train_label,test_feat,test_label = split_data(all_feat)
    test_data = Pool(data=test_feat, label=test_label['label'], cat_features = columns_c)

    # loading the model in the checkpoint
    May_model = May_cb()

    # test the given model
    test_model(train_feat,train_label,test_data,May_model)


