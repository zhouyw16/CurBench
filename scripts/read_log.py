import re
import os
import pandas as pd
from datetime import datetime
import shutil

cwd = os.getcwd()
path = cwd + '/runs'
cols = ['policy','data','net','epoch','seed','best valid acc','final test acc','duration','start datetime']
df = pd.DataFrame(columns=cols)

for d in os.listdir(path):
    contents = re.split('[_,-]',d)
    policy, data, net, epoch, seed = contents[2:7]
    dt = datetime.strptime(' '.join(contents[7:]), '%Y %m %d %H %M %S')
    cur_path = os.path.join(path,d)
    for file in os.listdir(cur_path):
        if file == 'train.log':
            with open(os.path.join(cur_path,file),'r') as f:
                lines = f.readlines()
                for line in lines:
                    mflag = re.search(r'Final Test Acc',line)
                    if mflag:
                        [best_valid, final_test]=re.findall("\d+\.\d+",line)
                        end_time = datetime.strptime("".join(re.findall(r'\d{4}-\d?\d-\d?\d (?:2[0-3]|[01]?[0-9]):[0-5]?[0-9]:[0-5]?[0-9]',line)), '%Y-%m-%d %H:%M:%S')
                        row = pd.DataFrame([contents[2:7] + [best_valid,final_test,end_time-dt,dt]],columns=cols)
                        print(row)
                        df = pd.concat([df,row],ignore_index=True)
df.reset_index()
df['duration'] = df['duration'].astype(str)
df.to_excel('scripts/results.xlsx')
        

# for d in os.listdir(path):
#     contents = re.split('[_,-]',d)
#     if 'bert' in contents:
#         cur_path = os.path.join(path,d)
#         shutil.rmtree(cur_path)