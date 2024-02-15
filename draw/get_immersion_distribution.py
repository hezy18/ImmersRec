import numpy as np
import pandas as pd
import json
from scipy.stats import pearsonr
    
foot_path = "../draw_model_result/"
# model_path = "WideDeep__KuaiRand_history_test_context101__0__lr=0.001__l2=0__emb_size=64__layers=[64].pt"
model_path = "WideDeep__KuaiRand_history__0__lr=0.001__l2=0__tradeoff_DA=0.0__tradeoff_Clf=0.0__emb_size=64__layers=[64].pt"
def save_source_dict(model_path):
    source_immersion = np.load(foot_path + model_path.split('.pt')[0]+'_source.npy')
    print(source_immersion.shape)
    # source_data = np.load(foot_path + model_path.split('.pt')[0]+'_source_data.npy')
    # print(source_data.shape)
    # print(source_immersion[0][0])

    source_csv_file_path = "source_data.csv"
    source_df = pd.read_csv('../draw_model_result/' + source_csv_file_path, sep='\t')
    print(source_df.head())
    print(len(source_df))

    source_dict={}
    for i in range(len(source_df)):
        user_id = str(source_df['user_id'][i])
        item_id = str(source_df['item_id'][i])
        rating_immerson = float(source_df['rating_immersion'][i])
        predict_immersion = float(source_immersion[i][0])
        if user_id not in source_dict:
            source_dict[user_id]={}
        source_dict[user_id][item_id]={'rating_immersion':rating_immerson,'predict_immersion':predict_immersion}


    source_label_file_path = "../SVRec.inter"
    source_label_df = pd.read_csv(source_label_file_path, sep='\t')
    print(source_label_df.head())
    print(len(source_label_df))

    for i in range(len(source_label_df)):
        user_id = str(source_label_df['user_id:token'][i])
        item_id = str(source_label_df['item_id:token'][i])
        rating_like = float(source_label_df['rating_like:float'][i])
        view_ratio = float(source_label_df['view_ratio:float'][i])
        if item_id in source_dict[user_id]:
            source_dict[user_id][item_id].update({'rating_like':rating_like,'view_ratio':view_ratio})

    with open(foot_path+'source_dict.json','w') as f:
        json.dump(source_dict,f)

save_source_dict(model_path)
        
def save_rec_csv(model_path):
    rec_df = pd.read_csv(foot_path + model_path.split('.pt')[0]+'.csv', sep='\t')
    print(rec_df.head())
    print(len(rec_df))
    
    import ast
    for column in ['rec_items','item_ids','predictions','immersions']:
        rec_df[column] = rec_df[column].apply(ast.literal_eval)
    rec_df['length'] = rec_df['item_ids'].apply(len)
    train_rec_df = rec_df.loc[rec_df['length'] == 2].copy()
    train_rec_df = train_rec_df.drop('length', axis=1)
    rec_df = rec_df.loc[rec_df['length'] != 2].copy()
    rec_df = rec_df.drop('length', axis=1)
    
    new_df = pd.DataFrame(train_rec_df['predictions'].tolist(), columns=['item_pos_score', 'item_neg_score'])
    new_df[['item_pos_immers', 'item_neg_immers']] = pd.DataFrame(train_rec_df['immersions'].tolist())
    train_rec_df = pd.concat([train_rec_df, new_df], axis=1)
    print(train_rec_df.head())
    
    train_rec_df['high_score_immers'] = np.where(train_rec_df['item_pos_score'] > train_rec_df['item_neg_score'], train_rec_df['item_pos_immers'], train_rec_df['item_neg_immers'])
    train_rec_df['low_score_immers'] = np.where(train_rec_df['item_pos_score'] < train_rec_df['item_neg_score'], train_rec_df['item_pos_immers'], train_rec_df['item_neg_immers'])
    
    def analysis():
        count_same_trend = (((train_rec_df['item_pos_score'] > train_rec_df['item_neg_score']) & 
                         (train_rec_df['item_pos_immers'] > train_rec_df['item_neg_immers'])) 
                        | ((train_rec_df['item_pos_score'] < train_rec_df['item_neg_score']) & 
                           (train_rec_df['item_pos_immers'] < train_rec_df['item_neg_immers']))).sum()
        count_more_immers = (train_rec_df['item_pos_immers'] >= train_rec_df['item_neg_immers']).sum()
        print('train',count_same_trend, len(train_rec_df), float(count_same_trend/len(train_rec_df)))
        print('train',count_more_immers, len(train_rec_df), float(count_more_immers/len(train_rec_df)))
        train_rec_df['compare_score'] = np.where(train_rec_df['item_pos_score'] > train_rec_df['item_neg_score'], 1, 0)
        train_rec_df['compare_immers'] = np.where(train_rec_df['item_pos_immers'] > train_rec_df['item_neg_immers'], 1, 0)
        print('train:immersion vs score - compare')
        print(train_rec_df[['compare_score', 'compare_immers']].corr())
        print(pearsonr(train_rec_df['compare_score'], train_rec_df['compare_immers']))
        print((train_rec_df['compare_immers']==1).sum())
    analysis()
    
    rec_df.to_csv(foot_path+'dev_test.csv', sep='\t')
    train_rec_df.to_csv(foot_path+'train.csv', sep='\t')

save_rec_csv(model_path)
    
def save_rec_dict_posneg():
    dev_test_df = pd.read_csv(foot_path+'dev_test.csv', sep='\t')
    train_df = pd.read_csv(foot_path+'train.csv', sep='\t')
    
    import ast
    for column in ['immersions']:
        dev_test_df[column] = dev_test_df[column].apply(ast.literal_eval)
    
    def all_immersion_dict():
        train_immersion = {col: train_df[col].tolist() for col in ['item_pos_immers','item_neg_immers']}
        devtest_immersion={'item_pos_immers':[],'item_neg_immers':[]}
        for index, row in dev_test_df.iterrows():
            devtest_immersion['item_pos_immers'].append(row['immersions'][0])
            devtest_immersion['item_neg_immers'].extend(row['immersions'][1:])
        print(len(devtest_immersion['item_pos_immers']),len(devtest_immersion['item_neg_immers']))
        rec_immers = {'train':train_immersion, 'devtest':devtest_immersion}
        with open(foot_path+'rec_immers_dict.json','w') as f:
            json.dump(rec_immers,f)
    all_immersion_dict()
            
    def user_immersion_dict():
        user_immersion={}
        user_dfs = train_df.groupby('user_id')
        for user_id, user_df in user_dfs:
            train_immersion = {col: user_df[col].tolist() for col in ['item_pos_immers','item_neg_immers']}
            user_immersion[user_id]={'train': train_immersion,'devtest':{'item_pos_immers':[],'item_neg_immers':[]}}
        for index, row in dev_test_df.iterrows():
            user_id = row['user_id']
            user_immersion[user_id]['devtest']['item_pos_immers'].append(row['immersions'][0])
            user_immersion[user_id]['devtest']['item_neg_immers'].extend(row['immersions'][1:])
        with open(foot_path+'rec_user_immers_dict.json','w') as f:
            json.dump(user_immersion,f)
    user_immersion_dict()
    
    def save_devtest_immers_nor():
        devtest_immersion={'item_pos_immers':[],'item_neg_immers':[]}
        for index, row in dev_test_df.iterrows():
            immers_nor = list(np.array(row['immersions']) - np.min(row['immersions'])) / (np.max(row['immersions']) - np.min(row['immersions']))
            devtest_immersion['item_pos_immers'].append(immers_nor[0])
            devtest_immersion['item_neg_immers'].extend(immers_nor[1:])
        with open(foot_path+'rec_devtest_immers_dict_nor.json','w') as f:
            json.dump(devtest_immersion,f)
    save_devtest_immers_nor()
    
save_rec_dict_posneg()

def save_rec_dict_highlow():
    train_df = pd.read_csv(foot_path+'train.csv', sep='\t')
    
    def user_immersion_dict():
        user_immersion_train={}
        score_immers_train={'high_score_immers':[],'low_score_immers':[]}
        user_dfs = train_df.groupby('user_id')
        for user_id, user_df in user_dfs:
            now_dict = {col: user_df[col].tolist() for col in ['high_score_immers','low_score_immers']}
            user_min = min(min(now_dict['high_score_immers']),min(now_dict['low_score_immers']))
            user_max = max(max(now_dict['high_score_immers']),max(now_dict['low_score_immers']))
            now_dict['high_score_immers'] = list((np.array(now_dict['high_score_immers']) - user_min) / (user_max - user_min))
            now_dict['low_score_immers'] = list((np.array(now_dict['low_score_immers']) - user_min) / (user_max - user_min))
            user_immersion_train[user_id] = now_dict
            score_immers_train['high_score_immers'].extend(now_dict['high_score_immers'])
            score_immers_train['low_score_immers'].extend(now_dict['low_score_immers'])
        with open(foot_path+'score_immers_train_nor.json','w') as f:
            json.dump(score_immers_train,f)  
    user_immersion_dict()
    
save_rec_dict_highlow()

def normalized_user_rec_result():
    with open(foot_path+'rec_user_immers_dict.json', 'r') as f:
        rec_user_immers_dict = json.load(f)
    immers={'item_pos_immers':[],'item_neg_immers':[]}
    for user in rec_user_immers_dict:
        # print(rec_user_immers_dict[user])
        # break
        user_min=np.inf
        user_max=-np.inf
        for group in rec_user_immers_dict[user]:
            for itemtype in rec_user_immers_dict[user][group]:
                if len(rec_user_immers_dict[user][group][itemtype])==0:
                    continue
                tmp_min = np.min(np.array(rec_user_immers_dict[user][group][itemtype],dtype=float))
                tmp_max = np.max(np.array(rec_user_immers_dict[user][group][itemtype],dtype=float))
                if tmp_min < user_min:
                    user_min = tmp_min 
                if tmp_max > user_max:
                    user_max = tmp_max
        for group in rec_user_immers_dict[user]:
            for itemtype in rec_user_immers_dict[user][group]:
                my_arr = np.array(rec_user_immers_dict[user][group][itemtype],dtype=float)
                nor_list = list((my_arr - user_min) / (user_max - user_min))
                rec_user_immers_dict[user][group][itemtype] = nor_list
                immers[itemtype].extend(nor_list)
    with open(foot_path+'rec_user_immers_dict_nor.json','w') as f:
        json.dump(rec_user_immers_dict,f)  
    with open(foot_path+'rec_user_immers_dict_nor_all.json','w') as f:
        json.dump(immers,f)  
        
normalized_user_rec_result()

def save_all_test_result(model_path):
    predictions = np.load(foot_path + model_path.split('.pt')[0]+'_all_predictions.npy')
    immersions = np.load(foot_path + model_path.split('.pt')[0]+'_all_immersions.npy')
    print(predictions.shape)
    print(immersions.shape)
    result = np.concatenate([predictions[:, 0, np.newaxis], immersions[:, 0, np.newaxis]], axis=1)
    print('all interaction: immersion vs score')
    print(pearsonr(result[:,0], result[:,1]))
    correlation_matrix = np.corrcoef(result, rowvar=False)
    print(correlation_matrix, correlation_matrix[0, 1])
    
    data_original = pd.read_csv('../src/data/KuaiRand_history_testall/all_original.csv',sep='\t')
    data_df = data_original[['user_id','session_id','c_session_order','label','session_length']]
    data_df['prediction'] = result[:, 0]
    data_df['immersion'] = result[:, 1]
    
    
    all_test_dict={} # user_id : {session_id: {session_order: [predictions,immersions,label,session_length]}
    user_dfs = data_df.groupby('user_id')
    for user_id, user_df in user_dfs:
        user_dict = user_df.groupby('session_id').apply(
                        lambda group: sorted(group.to_dict(orient='records'), key=lambda x: x['c_session_order'])
                        ).to_dict()
        all_test_dict[user_id]=user_dict

    with open(foot_path+'all_test_dict.json','w') as f:
        json.dump(all_test_dict,f)
    
save_all_test_result(model_path)