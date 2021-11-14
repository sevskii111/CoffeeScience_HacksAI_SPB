import json
import random
import pandas as pd
import numpy as np
import argparse
from tqdm.contrib.concurrent import thread_map

parser = argparse.ArgumentParser()
parser.add_argument('data_1', type=str,
                    help='train_1.csv like file')
parser.add_argument('data_2', type=str,
                    help='train_2.csv like file')
parser.add_argument('period_start', type=int,
                    help='First month to analyze')
parser.add_argument('period_len', type=int,
                    help='Amount of month to analyze')

args = parser.parse_args()

data_1 = pd.read_csv(args.data_1, delimiter=';', decimal=',', na_values=np.nan)
data_2 = pd.read_csv(args.data_2, delimiter=';', decimal=',', na_values=np.nan)

data_1[["MONTH", "YEAR"]] = data_1['TRANS_DTTM'].str.split(' ', expand=True).iloc[:, 0].str.split('.', expand=True).iloc[:, 1:].astype(int)

mcc_map_file = pd.read_csv('mccs.txt', index_col=0).to_dict('index')
unique_mcc_codes = data_1.MCC_CODE.unique()
mcc_map = {code: mcc_map_file[code]["Название"] for code in mcc_map_file if code in unique_mcc_codes}
mcc_name_map = {mcc_map_file[code]["Название"]: code for code in mcc_map_file if code in unique_mcc_codes}
data_1.MCC_CODE.replace(mcc_map, inplace=True)
unique_mcc_codes = mcc_map_file.keys()
mcc_map = {code: mcc_map_file[code]["Название"] for code in mcc_map_file if code in unique_mcc_codes}
mcc_name_map = {mcc_map_file[code]["Название"]: code for code in mcc_map_file if code in unique_mcc_codes}

grouped_transactions = data_1[['ID', 'MCC_CODE', 'SUM_TRANS', 'MONTH']].groupby(['ID', 'MCC_CODE', 'MONTH']).sum().reset_index()
grouped_transactions['ID'] = grouped_transactions['ID'].astype(int)

with open('groups.json') as groups_file:
    groups_json = json.load(groups_file)
groups = groups_json["Группы"]
adult_only = groups_json["18+"]
female_only = groups_json["F_ONLY"]
male_only = groups_json["M_ONLY"]

def get_preds_for(id, start_month=args.period_start, period=args.period_len):
    user = data_2[data_2.ID == id].iloc[0]
    user_transactions = grouped_transactions[(grouped_transactions.ID == id) & (grouped_transactions.MONTH <= start_month + period)].groupby('MCC_CODE').sum().reset_index()[["MCC_CODE", "SUM_TRANS"]].sort_values('SUM_TRANS', ascending=False)
    groups_to_draw = []
    for location in user_transactions["MCC_CODE"].values:
        transaction_group = None
        for group in groups:
            if location in groups[group]:
                transaction_group = group
                break
        if transaction_group is not None:
            groups_to_draw.append(transaction_group)
        
    top_5 = user_transactions["MCC_CODE"].values[:5]

    preds = []
    i = 0
    while len(preds) < 5:
        group_items = groups[groups_to_draw[i]] if i < len(groups_to_draw) else list(groups.items())[random.randint(0, len(groups) - 1)][1]
        possible_group_items = [item for item in group_items if (item not in top_5) and (item not in preds)]
        if user.GENDER == 1:
            map(possible_group_items.remove, female_only)
        else:
            map(possible_group_items.remove, male_only)
        if user.AGE < 18:
            map(possible_group_items.remove, adult_only)
        if len(possible_group_items) > 0:
            preds.append(random.choice(possible_group_items))
        i += 1
    #return top_5, preds
    return [id] + [mcc_name_map.get(mcc_name, mcc_name) for mcc_name in preds]

results = thread_map(get_preds_for, data_1.ID.unique())

pd.DataFrame(results, columns=['ID', 'REC_1', 'REC_2', 'REC_3', 'REC_4', 'REC_5']).to_csv('results.csv', index=None)