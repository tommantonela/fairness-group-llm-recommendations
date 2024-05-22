
import os
import pickle
import pandas as pd
import ranky

strat__ = ['min','avg','variance','all','max']
which__ = ['min','mean','median','max']

dir_path = './'
implicit_dir = dir_path + 'implicit/'

listi = []
for file in os.listdir(implicit_dir):
    
    if not file.startswith('gsumm'):
        continue
    
    print(file)
    impl = pd.read_pickle(implicit_dir + file)
    
    for strat in strat__:
        for which in which__:
            for i in range(0,len(impl)):
            
                if ('','') not in impl[i][10][strat]['novelty']:
                    continue
            
                listi.append({'name': f'{file}#{strat}_{which}',
                              'precision': impl[i][10][strat]['precision'][('','')][which],
                              'ndcg': impl[i][10][strat]['ndcg'][('','')][which],
                              'novelty': impl[i][10][strat]['novelty'][('','')][which]})

df_implicit = pd.DataFrame(listi)
df_implicit = df_implicit.drop_duplicates().groupby('name').mean()

with open(dir_path + '__df_implicit.pickle','wb') as ff:
    pickle.dump(df_implicit,ff)

aa = ranky.pairwise(df_implicit).to_dict()

with open(dir_path + '__ranky_results.pickle','wb') as ff:
    pickle.dump(aa,ff)