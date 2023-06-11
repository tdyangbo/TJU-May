import pandas as pd
data_path = 'data/feature/'

# loading feature
def load_all_feat(data_path):
    basic_feat = pd.read_csv(data_path+'basic_feat.csv')
    tags_feat = pd.read_csv(data_path+'alltags_feat.csv')
    title_feat = pd.read_csv(data_path+'title_feat.csv')
    caption_feat = pd.read_csv(data_path+'caption_feat.csv')
    graph_feat = pd.read_csv(data_path+'graph_feat.csv')
    all_data = pd.concat([basic_feat, tags_feat, title_feat, caption_feat, graph_feat], axis=1)
    return all_data

def split_data(all_data):
    train_all_data = all_data[all_data['train_type'] != -1]
    submit_all_data = all_data[all_data['train_type'] == -1]

    train_all_data = train_all_data.reset_index(drop=True)
    submit_all_data = submit_all_data.reset_index(drop=True)

    feature_columns = ['Pid', 'train_type', 'label', 'mean_label']
    feature_columns += ['user_fe_{}'.format(i) for i in range(399)]
    feature_columns += ['loc_fe_{}'.format(i) for i in range(400)]

    train_label_df = train_all_data[['Pid', 'label']]
    train_feature_df = train_all_data.drop(feature_columns, axis=1)

    submit_label_df = submit_all_data[['Pid', 'label']]
    submit_feature_df = submit_all_data.drop(feature_columns, axis=1)

    print(len(train_feature_df), len(submit_feature_df))
    print(len(train_label_df), len(submit_label_df))
    return  train_feature_df,train_label_df,submit_feature_df,submit_label_df

