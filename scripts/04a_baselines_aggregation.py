import pandas as pd
from datetime import datetime
import os
from tqdm.notebook import tqdm

from collections import Counter
import json
from collections import defaultdict
import pickle

def agg_additive(users,recs,to_recommend,**kwargs): 
    binary = kwargs.get('binary',False)
    recs_all = defaultdict(float)
    for u in users:
        for rr,vv in recs[u].items():
            recs_all[rr] += (vv if not binary else 1)
      
    recs_all = dict(sorted(recs_all.items(), key=lambda item: -item[1]))
    
    return [x for x in recs_all if x in to_recommend]

def agg_additive_bin(users,recs,to_recommend,**kwargs):
    return agg_additive(users,recs,to_recommend,binary=True)


import ranky
def borda(users,recs,to_recommend,**kwargs):
    recs_all = [v for k,v in recs.items() if k in users]
    recs_all = ranky.borda(pd.DataFrame(recs_all).T) 
    recs_all = recs_all.sort_values().index

    return [x for x in recs_all if x in to_recommend]


def least_misery(users,recs,to_recommend,**kwargs):
    recs_all = defaultdict(list)
    for u in users:
        for rr,v in recs[u].items():
            recs_all[rr].append(v)

    for rr,v in recs_all.items():
        recs_all[rr] = min(v) if kwargs.get('min_',True) else max(v)

    recs_all = dict(sorted(recs_all.items(), key=lambda item: -item[1]))

    return [x for x in recs_all if x in to_recommend]

def least_misery_max(users,recs,to_recommend,**kwargs):
    return least_misery(users,recs,to_recommend,min_=False)


dir_path = './'

df_movies = pd.read_csv(dir_path + 'ml-25m/movies.csv')
df_movies

df_movies_titles = {df_movies['movieId'].values[i] : df_movies['title'].values[i] for i in range(0,len(df_movies))}

aggregations = {'additive' : agg_additive,
                'additive-binary' : agg_additive_bin,
                'borda' : borda,
                'least-misery':least_misery,
                'least-misery-max':least_misery_max }

for ff in tqdm(os.listdir(dir_path)):
    
    if not ff.startswith('recommendations_implicit'):
        continue
    
    if not ff.endswith('.pickle'):
        continue

    if not os.path.exists(dir_path + '__'.join(ff.split('__')[1:])):
        print('Missing structure file for: ',ff)
        continue
    
    print(ff)
    
    recs = pd.read_pickle(dir_path + ff)
    structure = pd.read_pickle(dir_path + '__'.join(ff.split('__')[1:]))

    train_user_only = 5 if 'restricted_training' in ff else -1 # full_training
    test_user_only = 5 if 'restricted_training' in ff else -1
    nliked_user_only = 5 if 'restricted_training' in ff else -1

    for agg_name, agg in aggregations.items():
    
        final_recs = []
        
        path_out = 'results__implicit-' + agg_name + '_' + '__'.join(ff.split('__')[1:])
        
        if os.path.exists(dir_path + path_out):
            continue
        
        print('------',path_out)
        
        for group in tqdm(structure):
            sets_ = group[0]
            group_sets = group[1]
            users = set(group_sets.keys())

            to_recommend = sets_['test_intersect'] 
            for u in users:
                to_recommend.update(group_sets[u]['test_only'][:test_user_only if test_user_only != -1 else len(group_sets[u]['test_only'])])
                to_recommend.update(group_sets[u]['nliked_only'][:nliked_user_only if nliked_user_only != -1 else len(group_sets[u]['nliked_only'])])


            aggregation = agg(users,recs,to_recommend)
            aggregation = [df_movies_titles[x] for x in aggregation]

            final_recs.append([(('',''),json.dumps({'movies':aggregation}))]) 

        with open(dir_path + path_out,'wb') as file:
            pickle.dump(final_recs,file)