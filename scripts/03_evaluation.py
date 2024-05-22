import pandas as pd
from datetime import datetime
import os
from sklearn import metrics
from collections import deque
import pickle

class evaluation_metric():  

    def __init__(self):
        pass

    def get_scores(self, recs, **kwargs):
        pass

    def summarize(scores):
        summarized = {}
        summarized['min'] = min(scores, key=scores.get)
        summarized['min'] = (scores[summarized['min']],summarized['min'])
        summarized['avg'] = (np.mean(list(scores.values())),None) #sum(scores.values()) / np.std(list(scores.values()))
        summarized['avg_std'] = (summarized['avg'][0] - np.std(list(scores.values())),None)
        summarized['variance'] = (np.var(list(scores.values())),None)
        summarized['max'] = max(scores, key=scores.get)
        summarized['max'] = (scores[summarized['max']],summarized['max'])
        summarized['max-min'] = (summarized['max'][0] - summarized['min'][0],None)
        return summarized
    
class precision_k_user(evaluation_metric):

    def get_scores(self, recs, **kwargs): 
        ground_truth = kwargs['ground_truth']
        k = kwargs.get('k')
        k = k if k is not None else len(recs)
        k = min(k,len(recs))
        
        if len(recs[0:k]) > 0:
            scores = {u: sum(1 for x in recs[0:k] if x in ground_truth[u]) / k for u in ground_truth}
            summarized = evaluation_metric.summarize(scores)
            return scores,summarized
        
        return {}, {}

    
class ndcg_k_user(evaluation_metric):

    def get_scores(self, recs, **kwargs):

        ground_truth = kwargs['ground_truth']
        k = kwargs.get('k')
        k = k if k is not None else len(recs)
        k = min(k,len(recs))
        ndcgs = {}
        if len(recs[0:k]) <= 1:
            return ndcgs,{}
        
        for u in ground_truth:

            ground_truth_ = deque()
            recs_ = deque()

            for x in recs[0:k]:
                recs_.append(1)
                ground_truth_.append(1 if x in ground_truth[u] else 0)
            nd = metrics.ndcg_score([ground_truth_], [recs_])
            ndcgs[u] = nd
                   
        return ndcgs, evaluation_metric.summarize(ndcgs)

    
from sklearn.metrics.pairwise import cosine_distances

class distance__():

    def compute_distances_reduced(self, nodesA, nodesB):

        nodesA = nodesA.intersection(self.items.index) 
        nodesB = nodesB.intersection(self.items.index)

        if len(nodesA) == 0 or len(nodesB) == 0:
            return None, None, None

        matrixA = self.items.loc[sorted(list(nodesA))].astype(int)
        
        matrixB = self.items.loc[sorted(list(nodesB))].astype(int)
        
        node_indexes_row = {ke: v for v, ke in enumerate(matrixA.index)}
        node_indexes_columns = {ke: v for v, ke in enumerate(matrixB.index)}

        return cosine_distances(matrixA, matrixB), node_indexes_row, node_indexes_columns

class content_distance(distance__):

    def __init__(self, **kwargs):
        if 'items' in kwargs:
            self.items = kwargs['items']
        else:
            self.items_path = kwargs['items_path']
            self.items = self.load_representations(self.items_path, kwargs.get('euclidean', True))

    def load_representations(self, path, euclidean=True):
        print('Loading representations...', path)
        rep = pd.read_pickle(path)
        return rep
    
    
class individual_diversity(evaluation_metric):  
    def __init__(self, **kwargs):
        self.distance_ = kwargs['distance']

    def get_scores(self, recs, **kwargs):

        k = kwargs.get('k')
        k = k if k is not None else len(recs)
        k = min(k,len(recs))
        
        dists = {}
        
        if len(recs[0:k]) <= 1:
            return dists
        
        dd, node_indexes, _ = self.distance_.compute_distances_reduced(set(recs), set(recs))
        if dd is None:
            return dists
        
        dists['all'] = (dd.sum()-1).sum() / (len(dd) * (len(dd)-1)) # no need of a df as we sum all of them

        return dists


class individual_novelty(evaluation_metric):  

    def __init__(self, **kwargs):
        self.distance_ = kwargs['distance']

    def get_scores(self, recs, **kwargs):

        k = kwargs.get('k')
        k = k if k is not None else len(recs)
        k = min(k,len(recs))
        
        if len(recs[0:k]) > 0:
            known = kwargs['known']
            recs_set = set(recs)
            dists = {}
            for u, gt in known.items():  # for each user

                dd, node_indexes_row, node_indexes_columns = self.distance_.compute_distances_reduced(recs_set, set(gt))
                if dd is None:
                    continue

                dists[u] = dd.sum().sum() / (dd.shape[0] * dd.shape[1]) 
#             print(dists)
            return dists, evaluation_metric.summarize(dists)
        
        return {}, {}
    
    
from collections import Counter
import numpy as np
class coverage(evaluation_metric): 
        
    def get_scores(self, recs, **kwargs):
        
        ground_truth = kwargs['ground_truth']
        nliked_users = kwargs['nliked_users']
        k = kwargs.get('k')
        k = k if k is not None else len(recs)
        k = min(k,len(recs))
        
        anti = {}
        
        if len(recs[0:k]) < 1:
            return anti,anti
        
        # tres métricas de una
        anti['coverage_relevants'] = Counter()
        anti['coverage_not_relevants'] = Counter()
        anti['coverage_unknown'] = Counter()
        
        for u in ground_truth:

            for x in recs[0:k]:
                if x in ground_truth[u]:
                    anti['coverage_relevants'][u] += 1
                elif x in nliked_users[u]:
                    anti['coverage_not_relevants'][u] += 1
                else:
                    anti['coverage_unknown'][u] += 1
            
            anti['coverage_relevants'][u] = anti['coverage_relevants'][u] / k
            anti['coverage_not_relevants'][u] = anti['coverage_not_relevants'][u] / k
            anti['coverage_unknown'][u] = anti['coverage_unknown'][u] / k
            
        summarized = {}
        summarized['coverage_relevants'] = evaluation_metric.summarize(anti['coverage_relevants']) 
        summarized['coverage_not_relevants'] = evaluation_metric.summarize(anti['coverage_not_relevants']) 
        summarized['coverage_unknown'] = evaluation_metric.summarize(anti['coverage_unknown']) 
         
        return anti, summarized
    
    
def compute_metrics(ground_truth, train, nliked_users, recommendations_valid, recommendations,k=None): 
    results = {}
    summarized_results = {}

    for m,metric in metrics_.items():
        if m != 'coverage':
            results[m], summarized_results[m] = metric.get_scores(recommendations_valid,ground_truth=ground_truth,known=train,nliked_users=nliked_users,k=k)
        else:
            rr = metric.get_scores(recommendations_valid,ground_truth=ground_truth,known=train,nliked_users=nliked_users,k=k)
            results.update(rr[0])
            summarized_results.update(rr[1])
        
    all_ground = set().union(*[v for v in ground_truth.values()])
    all_nliked = {'all' : set().union(*[v for v in nliked_users.values()])}
    
    for x in ['precision','ndcg']:
        if x not in results:
            continue
        
        results[x].update(metrics_[x].get_scores(recommendations,ground_truth={'good':recommendations_valid},k=k)[0])
        results[x].update(metrics_[x].get_scores(recommendations_valid,ground_truth={'all':all_ground},k=k)[0])
        if 'good' in results:
            summarized_results[x]['good'] = (results[x]['good'],None)
        if 'all' in results[x]:
            summarized_results[x]['all'] = (results[x]['all'],None)
        
    if 'coverage_relevants' in results:
        rr = metrics_['coverage'].get_scores(recommendations,ground_truth={'all':all_ground},nliked_users=all_nliked,k=k)[0]
        for mm in rr:
#             print(rr[mm])
            results[mm]['all'] = rr[mm]['all']
            summarized_results[mm]['all'] = (rr[mm]['all'],None) 
            
    results['diversity'] = (diversity.get_scores(recommendations_valid,k=k), None) # no requiere nada más

    return results,summarized_results


from tqdm import tqdm
import json
import re

pp = re.compile("```(json|){(.|\n)*}```")
pp2 = re.compile("{([^}]*)\}")

re_missing_brace = re.compile('{.*]')

re_comments = re.compile('(\/\/|##).*\n')

re_mutiple_lines = re.compile('\n{2,}')
re_check_extra_comma = re.compile('(],* *){2,}')
re_check_ending = re.compile('(] *){2,}]{1,}')

re_inbetween_quotes = re.compile('"([^"]+)"')

def ground_truths(users, to_recommend, base_i):
    gt = {}
    train = {}
    not_liked = {}
    for x in users:
        test_x = set(base_i[1][x]['test_user'])
        gt[x] = test_x.intersection(to_recommend)  
        train[x] = set(base_i[1][x]['train_user'])
        not_liked[x] = set(base_i[1][x]['nliked_only'])
    return gt, train, not_liked
    
def matching(x,y):
    
    if y is None:
        return None
    
    if '(' in x:
        xs = x.split('(')
        xs = '('.join(xs[0:-1])
    else: 
        xs = x
    
    return y if y.startswith(xs) else None

def parse_movies(rr):
            
    if 'unable to generate' in rr:
        return []
    
    if '//' in rr or '##' in rr:
        rr = re_comments.sub('',rr)

    aa = pp.search(rr)
    if aa is None:
        aa = pp2.search(rr)

    aa = aa[0] if aa is not None else aa

    if aa is None:

        xx = re_mutiple_lines.search(rr)
        if xx is not None: # TODO !!!!!
            rr = rr[0:xx.span()[0]]
            rr = rr.replace('\n',' ').strip()

            rr = re_check_extra_comma.sub(']]',rr)
            if re_check_ending.search(rr):
                aa = rr[0:-1] + '}'
    else:
        aa = aa[3:-3] if aa.startswith('```') else aa 
        aa = aa if not aa.startswith('json') else aa[4:]

    if aa is None:
        rr = rr.replace('`','')

        if '{' in rr and '}' not in rr:
            aa = rr + '}'

    aa = aa[:aa.index('[')+1] + aa[aa.index('[')+1:].replace('[','').replace(']','')
    aa = aa[:aa.rindex('}')] + ']}'

    if aa == "{'movies': []}": 
        aa = aa.replace('\'','"') 

    try: 
        recs = json.loads(aa)['movies']
    except Exception as e: 
        recs = re_inbetween_quotes.findall(aa)
       

    # Check genres
    recs = [x['title'] if isinstance(x,dict) else x for x in recs if x != 'movies' and x != 'title' and x != 'rank' and not isinstance(x,int) and not isinstance(x,float) and 'rime' not in x and 'antasy' not in x] # este venía funcionando bien, pero ya no es el caso
    return recs


def compute_group_results(results,groups,base):

    group_results = []

    for i in tqdm(range(0,len(results))): 
        results_i = results[i]
        base_i = base[i]
        group_i = groups[i]

        users = list(base_i[1].keys()) # users in group_i

        to_recommend_ids = set(df_movies[df_movies['title'].isin(group_i['to_recommend'])]['movieId'].values)

        gt_users, train_users, nliked_users = ground_truths(users, to_recommend_ids, base_i)

        stereotype_results = []

        for j in tqdm(range(0,len(results_i))): 

            st_res = {}

            stereotype_j = results_i[j]
            rr = stereotype_j[1] if isinstance(stereotype_j[1],str) else stereotype_j[1].content
            recs = parse_movies(rr)

            st_res['recommended'] = recs 
            st_res['recommended_ids'] = [movie_titles_set[x] if x in movie_titles_set else movie_titles_set[next((y for y in movie_titles_set if matching(x, y)), None)] for x in recs]

            st_res['recs_valid'] = [match for x in recs for y in group_i['to_recommend'] if (match := matching(x, y)) is not None]
            st_res['recs_valid_ids'] = [movie_titles_set[x] for x in st_res['recs_valid']]
            st_res['extra'] = [st_res['recommended'][i] for i in range(0,len(st_res['recommended'])) if st_res['recommended_ids'][i] == -1] 

            st_res['metrics'] = {}
            st_res['summarized_metrics'] = {}

            st_res['metrics'][-1], st_res['summarized_metrics'][-1] = compute_metrics(gt_users, train_users, nliked_users, st_res['recs_valid_ids'],st_res['recommended_ids'],k=None)
            st_res['metrics'][1], st_res['summarized_metrics'][1] = compute_metrics(gt_users, train_users, nliked_users, st_res['recs_valid_ids'],st_res['recommended_ids'],k=1)
            st_res['metrics'][5], st_res['summarized_metrics'][5] = compute_metrics(gt_users, train_users, nliked_users, st_res['recs_valid_ids'],st_res['recommended_ids'],k=5)
            st_res['metrics'][10], st_res['summarized_metrics'][10] = compute_metrics(gt_users, train_users, nliked_users, st_res['recs_valid_ids'],st_res['recommended_ids'],k=10)

            st_res['stereotype_id'] = None if group_i['stereo_combinations'][j][1] is None else int(group_i['stereo_combinations'][j][1].split()[2])
            
            st_res['#users'] = len(users)

            stereotype_results.append((stereotype_j[0],st_res)) 
        group_results.append(stereotype_results)
        
    return group_results


# CORRELATION ANALYSIS
import numpy as np
import scipy
from collections import defaultdict

def average_correlations(coeffs):
    z_transformed = [0.5 * np.log((1 + (r[0] if not isinstance(r, float) else r)) / (1 - (r[0] if not isinstance(r, float) else r))) for r in coeffs if (r[0] if not isinstance(r, float) else r) != 1] # Apply Fisher's Z transformation
    if len(z_transformed) == 0: # all was 1.0
        return 1.0    
    average_z = np.mean(z_transformed)
    average_correlation = (np.exp(2 * average_z) - 1) / (np.exp(2 * average_z) + 1)
    return average_correlation


def compute_group_correlations(group_results):

    group_correlations = [] # [(all,same_user)]
    avg_group_correlations = [] # [(all,same_user)]

    for stereotype_results in tqdm(group_results):
        correlations = defaultdict(defaultdict(list).copy) 
        correlations_same = defaultdict(defaultdict(list).copy)
        for i in range(0,len(stereotype_results)): 
            ste_i = stereotype_results[i]
            rec_i = ste_i[1]['recommended_ids']
            for j in range(0,len(stereotype_results)):

                if i == j:
                    continue

                ste_j = stereotype_results[j]
                rec_j = ste_j[1]['recommended_ids']

                for k in ks:
                    k = min(k,len(rec_i),len(rec_j))
                    if k <= 1:
                        continue   
   
                    co = scipy.stats.kendalltau(rec_i[0:k],rec_j[0:k])
   
                    correlations[k][(ste_i[0], ste_j[0])].append(co) 

                    if ste_i[1]['stereotype_id'] is None or ste_j[1]['stereotype_id'] is None or ste_i[1]['stereotype_id'] == ste_j[1]['stereotype_id']:
    #                 
                        correlations_same[k][(ste_i[0], ste_j[0])].append(co)

        group_correlations.append((correlations,correlations_same))
        
        avg_group_correlations.append(({k : {ss:average_correlations(v) for ss,v in correlations[k].items()} for k in correlations},
                                       {k : {ss:average_correlations(v) for ss,v in correlations_same[k].items()} for k in correlations_same}))
        
    return group_correlations, avg_group_correlations


def correlation_avg_accros_groups(avg_group_correlations,which=1):
    full_avg = defaultdict(defaultdict(list).copy)
    full_avg_final = defaultdict(defaultdict(defaultdict(int).copy).copy)
    
    for k in avg_group_correlations[0][which]:
        for i in range(0,len(avg_group_correlations)):
            for ste,v in avg_group_correlations[i][which][k].items():
                full_avg[k][ste].append(v)
        for ste in full_avg[k]:
            full_avg_final[k][' '.join(ste[0])][' '.join(ste[1])] = average_correlations(full_avg[k][ste])
    return full_avg_final


from collections import Counter, defaultdict

def compute_group_all_summarized(group_results):

    group_all_summarized = []

    for j in tqdm(range(0,len(group_results))):
        stereotype_results = group_results[j]
        acumm_metrics = defaultdict(defaultdict(defaultdict(defaultdict(list).copy).copy).copy) # {k : {acum_strategy : {metric: {(stereotype): [scores]} }}}


        rels = defaultdict(defaultdict(defaultdict(Counter).copy).copy) # {k : acum_strategy : {metric : {stereotype : counter}}}
        count_stere = defaultdict(defaultdict(defaultdict(Counter).copy).copy)
        for i in range(0,len(stereotype_results)): 
            ste_i = stereotype_results[i]
            metrics_i = ste_i[1]['summarized_metrics']
            ste_i_id = ste_i[1]['stereotype_id']

            for k, scores in metrics_i.items(): 

                for m, mm in scores.items(): 
                    for aa, sc in mm.items():

                        acumm_metrics[k][aa][m][ste_i[0]].append(sc[0])

                        acumm_metrics[k][aa][m][(ste_i[0][0],None)].append(sc[0]) 
                        acumm_metrics[k][aa][m][(None,ste_i[0][1])].append(sc[0]) 

                        if sc[1] is not None and ste_i_id == sc[1]: 
                            rels[k][aa][m][ste_i[0]] += 1
                        count_stere[k][aa][m][ste_i[0]] += 1

        for k, strat in rels.items(): 
            for st, mm_scores in strat.items():
                for metric, count in mm_scores.items():
                    tot = sum(count.values())
                    for ss, v in count.items():
                        acumm_metrics[k][f'rel_{st}'][metric][ss].append(v/count_stere[k][st][metric][ss] if count_stere[k][st][metric][ss] > 0 else 0)

        gas = defaultdict(defaultdict(defaultdict(dict).copy).copy) # {k : {ww : mm : []}
        for k,strat in acumm_metrics.items():
            for ww, me in strat.items():
                for mm, stereotypes in me.items():
                    for st, v in stereotypes.items():
    #                     print(k,ww,mm,st,v)
                        gas[k][ww][mm][st] = {'min': np.min(v), 'mean':np.mean(v), 'std':np.std(v),'median':np.percentile(v,50),'max':np.max(v)}

        group_all_summarized.append(gas)
    return group_all_summarized


def extract_group_x_stereotype(group_all_summarized,k=10,strat='avg',which='mean',metric=None):
    
    if metric is None:
        metric = ['precision','ndcg','novelty']

    if isinstance(metric,str):
        metric = [metric]

    stereotypes = set()
    for i in range(0,len(group_all_summarized)):
        for m in metric:
            stereotypes.update(group_all_summarized[i][k][strat][m].keys())

    mapping_names = {x : str(x[0]) + ' ' + str(x[1]) for x in stereotypes}

    groups_x_stereotype = {m : {mapping_names[x]:[] for x in stereotypes} for m in metric} 
    groups_x_stereotype_list = []

    for i in range(0,len(group_all_summarized)):

        for m in metric:

            if len(group_all_summarized[i][k][strat][m]) == 0:
                continue

            for x in stereotypes:
                xx = group_all_summarized[i][k][strat][m].get(x,np.nan) 
                xx = xx if xx != xx else xx[which]
                groups_x_stereotype[m][mapping_names[x]].append(xx)
                groups_x_stereotype_list.append({'metric':m,'strat':strat,'which':which,'stereotype':mapping_names[x],'value':xx})
            
    return groups_x_stereotype, groups_x_stereotype_list


def compute_user_representation(group_results,metric_type='metrics',k=10,use_orig_value=False):

    group__users_x_metrics_x_alt_x_value = []
    
    for stereotype_results in group_results:

        users_x_metrics_x_alt_x_value = defaultdict(defaultdict(defaultdict(defaultdict(float).copy).copy).copy)

        for i in tqdm(range(0,len(stereotype_results))): 
            results_i = stereotype_results[i]

            stereotype_id = results_i[1]['stereotype_id']

            for mm, mm_scores in results_i[1][metric_type][k].items():

                if mm == 'diversity':
                    continue

                for ss, v in mm_scores.items():   
                    
                    if ss == 'avg' or ss == 'avg_std' or ss == 'variance' or ss == 'max-min': 
                        continue
                                                               
                    if metric_type == 'summarized_metrics':
                       
                        if ss == 'good' or ss == 'all':
                            users_x_metrics_x_alt_x_value[mm][ss][ss][' '.join(results_i[0])] = 1 if not use_orig_value else v[0]
                            continue 

                        if stereotype_id is None or v[1] == stereotype_id:
                            users_x_metrics_x_alt_x_value[mm][ss][v[1]][' '.join(results_i[0])] = 1 if not use_orig_value else v[0]
                        else:
                            users_x_metrics_x_alt_x_value[mm][ss][v[1]][' '.join(results_i[0]) + '_others'] += (1 if not use_orig_value else v[0]) # TODO: Este no es uno solo, son varios!
                                
                    else: 
                        
                        if ss == 'good' or ss == 'all':
                            users_x_metrics_x_alt_x_value[mm]['user'][ss][' '.join(results_i[0])] = v
                            continue 

                        if stereotype_id is None or ss == stereotype_id:
                            users_x_metrics_x_alt_x_value[mm]['user'][ss][' '.join(results_i[0])] = v
                        else:
                            users_x_metrics_x_alt_x_value[mm]['user'][ss][' '.join(results_i[0]) + '_others'] += v

        for mm in users_x_metrics_x_alt_x_value:
            for ss in users_x_metrics_x_alt_x_value[mm]:
                for uu in users_x_metrics_x_alt_x_value[mm][ss]:
                    for x in users_x_metrics_x_alt_x_value[mm][ss][uu]:
                        if not x.endswith('_others'):
                            continue
                        users_x_metrics_x_alt_x_value[mm][ss][uu][x] = users_x_metrics_x_alt_x_value[mm][ss][uu][x] / (results_i[1]['#users'] - 1)
        group__users_x_metrics_x_alt_x_value.append(users_x_metrics_x_alt_x_value) 
    
    return group__users_x_metrics_x_alt_x_value

# --------------------------------------------------------

dir_path = './'

df_movies = pd.read_csv(dir_path + 'ml-25m/movies.csv')
df_movies

if not os.path.exists(dir_path + 'ml-25m\movies_genres.pickle'):
    df_movies['genres_'] = [x.split('|') for x in df_movies['genres']]
    df_movies__ = df_movies.explode('genres_')
    df_movies__['a'] = 1
    df_movies__ = df_movies__.reset_index(drop=True)
    df_movies__ = df_movies__.drop_duplicates()
    df_movies_genre = df_movies__.pivot(index='movieId',columns='genres_',values='a').fillna(0).astype(int)
    del df_movies__
    df_movies_genre.to_pickle(dir_path + 'ml-25m\movies_genres.pickle')
else:
    df_movies_genre = pd.read_pickle(dir_path + 'ml-25m\movies_genres.pickle')

metrics_ = {}
metrics_['precision'] = precision_k_user()
metrics_['ndcg'] = ndcg_k_user()
metrics_['coverage'] = coverage()
cdd = content_distance(items_path=dir_path + 'ml-25m/movies_genres.pickle')
diversity = individual_diversity(distance=cdd)
metrics_['novelty'] = individual_novelty(distance=cdd)


movie_titles_set = {df_movies['title'].values[i] : df_movies['movieId'].values[i] for i in range(0,len(df_movies))}
movie_titles_set[None] = -1

ks = [-1,1,5,10]


import os

for file in tqdm(os.listdir(dir_path)):

    if not file.startswith('results__'):
        continue
    
    print('--------------------------------',file)
    
    results = pd.read_pickle(dir_path + file)
    groups = '__'.join(file.split('__')[2:])
    base ='__'.join(groups.split('__')[:-1]) + '.pickle'
    
    groups = pd.read_pickle(dir_path + groups)
    base = pd.read_pickle(dir_path + base)
    
    if len(results) != len(groups): 
        continue


    group_results_file = 'groupr__' + file 
    print('Computing group results...',str(datetime.now())) 
    if not os.path.exists(dir_path + group_results_file):
        group_results = compute_group_results(results,groups,base)
        with open(dir_path + group_results_file,'wb') as ff:
            pickle.dump(group_results,ff)
    else: 
        group_results = pd.read_pickle(dir_path + group_results_file)
    
    group_correlations_file = 'corr__' + file
    print('Computing group correlations...',str(datetime.now()))
    if not os.path.exists(dir_path + group_correlations_file):
        group_correlations, avg_group_correlations = compute_group_correlations(group_results)
        full_avg_final = correlation_avg_accros_groups(avg_group_correlations,which=1)
        with open(dir_path + group_correlations_file,'wb') as ff:
            pickle.dump([group_correlations, avg_group_correlations,full_avg_final],ff)

    group_summarized_file = 'gsumm__' + file
    print('Computing compute_group_all_summarized...',str(datetime.now()))
    if not os.path.exists(dir_path + group_summarized_file):
        group_all_summarized = compute_group_all_summarized(group_results)
        with open(dir_path + group_summarized_file,'wb') as ff:
            pickle.dump(group_all_summarized,ff)
    else:
        group_all_summarized = pd.read_pickle(dir_path + group_summarized_file)
    
    print('Computing extract_group_x_stereotype',str(datetime.now()))
    group_x_stereotype_file = 'ge_10_avg_mean__' + file
    if not os.path.exists(dir_path + group_x_stereotype_file): 
        groups_x_stereotype, groups_x_stereotype_list = extract_group_x_stereotype(group_all_summarized,k=10,strat='avg',which='mean')
        with open(dir_path + group_x_stereotype_file,'wb') as ff:
            pickle.dump([groups_x_stereotype, groups_x_stereotype_list],ff)
    
    print('Computing compute_user_representation',str(datetime.now()))
    user_representation_file = 'user_mm_10_F__' + file
    if not os.path.exists(dir_path + user_representation_file): 
        group__users_x_metrics_x_alt_x_value = compute_user_representation(group_results,metric_type='metrics',k=10,use_orig_value=False)
        with open(dir_path + user_representation_file,'wb') as ff:
            pickle.dump(group__users_x_metrics_x_alt_x_value,ff)
    user_representation_file = 'user_smm_10_F__' + file
    if not os.path.exists(dir_path + user_representation_file): 
        group__users_x_metrics_x_alt_x_value = compute_user_representation(group_results,metric_type='summarized_metrics',k=10,use_orig_value=False)
        with open(dir_path + user_representation_file,'wb') as ff:
            pickle.dump(group__users_x_metrics_x_alt_x_value,ff)
    user_representation_file = 'user_mm_10_T__' + file
    if not os.path.exists(dir_path + user_representation_file): 
        group__users_x_metrics_x_alt_x_value = compute_user_representation(group_results,metric_type='metrics',k=10,use_orig_value=True)
        with open(dir_path + user_representation_file,'wb') as ff:
            pickle.dump(group__users_x_metrics_x_alt_x_value,ff)
    user_representation_file = 'user_smm_10_T__' + file
    if not os.path.exists(dir_path + user_representation_file): 
        group__users_x_metrics_x_alt_x_value = compute_user_representation(group_results,metric_type='summarized_metrics',k=10,use_orig_value=True)
        with open(dir_path + user_representation_file,'wb') as ff:
            pickle.dump(group__users_x_metrics_x_alt_x_value,ff)