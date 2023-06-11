import numpy as np
import pandas as pd
import json

def save_submission(test_proba,test_label):
    test_ans = np.mean(test_proba, axis=0)
    result = pd.DataFrame()
    result['post_id'] = test_label['Pid'].apply(lambda x: 'post' + str(x))
    result['popularity_score'] = test_ans.round(decimals=4)
    submit_data = dict()
    submit_data["version"] = "VERSION 1.2"
    submit_data["result"] = result.to_dict(orient='records')
    submit_data["external_data"] = {"used": "true", "details": "VGG-19 pre-trained on ImageNet training set"}

    file = open('results/result.json', "w")
    json.dump(submit_data, file)
    file.close()


def save_checkpoint_result(test_proba, k):
    test_ans = np.mean(test_proba, axis=0)
    result = pd.DataFrame()
    result['post_id'] = test_label_df['Pid'].apply(lambda x: 'post' + str(x))
    result['popularity_score'] = test_ans.round(decimals=4)

    checkpint_result = dict()
    checkpint_result["version"] = "VERSION 1.2"
    checkpint_result["result"] = result.to_dict(orient='records')
    checkpint_result["external_data"] = {"used": "true", "details": "VGG-19 pre-trained on ImageNet training set"}
    file = open('results/result_%s.json' % k, "w")
    json.dump(checkpint_result, file)
    file.close()