import re
import os
import pandas as pd
from datetime import datetime
import shutil

cwd = os.getcwd()
path = cwd + '/runs'
cols = ['policy','data','noise', 'net','epoch','seed','best valid acc','final test acc']
df = pd.DataFrame(columns=cols)

for d in os.listdir(path):
    contents = re.split('[_,-]',d)

    if 'noise' in contents :
        policy, data, noise, level, net, epoch, seed = contents[2:]
        noise_level = level
    elif 'imbalance' in contents:
        policy, data, imbalance, level, net, epoch, seed = contents[2:]
        noise_level = imbalance + level
    else:
        policy, data, net, epoch, seed = contents[2:]
        noise_level = 0

    cur_path = os.path.join(path,d)
    for file in os.listdir(cur_path):
        if file == 'train.log':
            with open(os.path.join(cur_path,file),'r') as f:
                lines = f.readlines()
                for line in lines:
                    final_flag = re.search(r'Final Test Acc',line)
                    best_flag = re.search(r'^\d{4}-\d?\d-\d?\d (?:2[0-3]|[01]?[0-9]):[0-5]?[0-9]:[0-5]?[0-9]\tBest Valid \w+ = \d+\.\d+',line)

                    if final_flag:
                        [best_valid, final_test]=re.findall("\d+\.\d+",line)
                        # end_time = datetime.strptime("".join(re.findall(r'\d{4}-\d?\d-\d?\d (?:2[0-3]|[01]?[0-9]):[0-5]?[0-9]:[0-5]?[0-9]',line)), '%Y-%m-%d %H:%M:%S')
                        row = pd.DataFrame([[policy,data,noise_level,net,epoch,seed,best_valid,final_test]],columns=cols)  # 8 cloumns
                        # print(row)
                        df = pd.concat([df,row],ignore_index=True)

                    elif best_flag:
                        best_valid = re.findall("\d+\.\d+",line)[0]
                        final_test = ''
                        # end_time = datetime.strptime("".join(re.findall(r'\d{4}-\d?\d-\d?\d (?:2[0-3]|[01]?[0-9]):[0-5]?[0-9]:[0-5]?[0-9]',line)), '%Y-%m-%d %H:%M:%S')
                        row = pd.DataFrame([[policy,data,noise_level,net,epoch,seed,best_valid,'']],columns=cols)  # 8 cloumns
                        # print(row)
                        df = pd.concat([df,row],ignore_index=True)

                    

df.reset_index()
df.to_excel('scripts/results33.xlsx')
        

# for d in os.listdir(path):
#     contents = re.split('[_,-]',d)
#     if 'bert' in contents:
#         cur_path = os.path.join(path,d)
#         shutil.rmtree(cur_path)