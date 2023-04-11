from itertools import combinations
from math import comb
from typing import Tuple, List
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd
import numpy as np
import multiprocessing

def calculate_score(model, combo, X, target, crossval, scoring):
    scores = cross_val_score(model, X[list(combo)], target, cv=crossval, scoring=scoring)
    r2 = scores.mean()
    return (r2, combo, X.columns)

### THIS IS THE FASTEST METHOD###
def bestFeatures(data: pd.DataFrame, target: pd.Series, crossval, model, n_features: int = 5, scoring: str = 'r2', n_jobs: int = None) -> Tuple[float, List[str], int, List[float], List[List[str]]]:
    """Tool to test all of possible combinations features list_vars
    WARNING: THIS IS A VERY SLOW PROCESS, USE IT ONLY FOR SMALL DATASETS
    
    
    By Helder Fraga helderfraga@gmail.com
    
    Example of use:
    from Utils import bestFeatures

    X = data.read_csv('data.csv')
    X = X.drop('target',axis=1)
    y = data['target']

    model = RandomForestClassifier(random_state=42)
    skf = KFold(n_splits=5, random_state=42, shuffle=True)

    max_score, max_vars, max_nr,score,cols = bestFeatures (X, y, skf, model, 5, scoring='recall', n_jobs=4)

    
    data are the dataframe with the features
    target is a series
    crossval is a cross validation method
    model is a model
    n_features is a number for the number features to be combined
    
    example data:
    data = pd.read_csv('data.csv')
    target = data['target']   
    crossval = KFold(n_splits=5,shuffle=True,random_state=42)
    model = RandomForestRegressor(n_estimators=100,random_state=42)
    n_features = 10
    n_jobs = -1
    

    """
    print('starting')
    max_score = 0
    score = []
    cols = []
    max_vars = []
    nr = 0
    list_vars = data.columns.tolist()
    print(comb(len(list_vars), n_features), 'combinations')

    
    # create a list of all possible combinations
    combos = list(combinations(list_vars, n_features))

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()-1
    if n_jobs == 0:
        n_jobs = 1
    else:
        n_jobs = n_jobs     

    # create a pool of worker processes
    pool = multiprocessing.Pool(processes=n_jobs)
    
    # map the calculation of each score to the worker processes
    results = [pool.apply_async(calculate_score, args=(model, combo, data, target, crossval, scoring)) for combo in combos]
    
    # get the results and update the maximum score
    for r in results:
        nr += 1
        r2, combo, cols_combo = r.get()
        score.append(r2)
        cols.append(cols_combo)
        #print(nr)

        if r2 > max_score:
            max_score = r2
            max_vars = combo
            max_nr = nr
            print(max_score, max_vars, max_nr, model.__class__.__name__)

    pool.close()
    pool.join()
    
    return (max_score, max_vars, max_nr, score, cols)




def calc_score(idx: List[int], X, target, model, crossval, scoring):
    X_subset = X.iloc[:, idx]
    scores = cross_val_score(model, X_subset, target, cv=crossval, scoring=scoring)
    return scores.mean()

### THIS IS THE SLOWEST METHOD###
def bestFeatures_parallel_slower(data: pd.DataFrame, target: pd.Series, crossval: KFold, model, n_features: int = 5, scoring: str = 'r2', n_jobs: int = 4) -> Tuple[int, List[str], int, List[float], List[List[str]]]:
    print('starting')
    max_score = 0
    score = []
    cols = []
    max_vars = []
    nr = 0
    list_vars = data.columns.tolist()
    print(comb(len(list_vars), n_features), 'combinations')


    with multiprocessing.Pool(n_jobs) as p:
        results = []
        for i in range(n_features, n_features + 1):
            for combo in combinations(list_vars, i):
                nr += 1
                cols.append(combo)
                idx_list = [data.columns.get_loc(c) for c in combo]
                results.append(p.apply_async(calc_score, (idx_list, data, target, model, crossval, scoring)))
        for r in results:
            r.wait()
            r_value = r.get()
            score.append(r_value)
            if r_value > max_score:
                max_score = r_value
                max_vars = cols[results.index(r)]
                max_nr = nr
                print(max_score, max_vars, max_nr)

    return max_score, list(max_vars), max_nr, score, cols


### OLD METHOD ###
def bestFeatures_old (data, target,crossval,model,n_features=5,scoring='r2') -> Tuple[int, list, int, list,list]:
    from itertools import combinations
    from sklearn.model_selection import cross_val_score
    from math import comb

    """Tool to test all of possible combinations features list_vars
    WARNING: THIS IS A VERY SLOW PROCESS, USE IT ONLY FOR SMALL DATASETS
    ABOUT 5 MINUTES FOR 1200 COMBINATIONS IN A RECENT CPU
    
    By Helder Fraga helderfraga@gmail.com
    
    Example of use:
    from Utils import bestFeatures

    X = data.read_csv('data.csv')
    X = X.drop('target',axis=1)
    y = data['target']

    model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0,class_weight='balanced')
    skf = KFold(n_splits=5, random_state=42, shuffle=True)

    max_score, max_vars, max_nr,score,cols = bestFeatures (X, y, skf, model, 5, scoring='recall')

    
    data is a dataframe
    target is a series
    crossval is a cross validation method
    model is a model
    n_features is a number for the number features to be combined
    
    example data:
    data = pd.read_csv('data.csv')
    target = data['target']   
    crossval = KFold(n_splits=5,shuffle=True,random_state=42)
    model = RandomForestRegressor(n_estimators=100,random_state=42)
    n_features = 10
    
    example call:
    from Utils import Best_Features
    max_score, max_vars, max_nr,score,cols = Best_Features (data, target, crossval, model, 10)
    """

    print ('starting')
    max_score = 0
    score = []
    cols = []
    max_vars = []
    max_nr = 0
    nr = 0
    list_vars = data.columns.tolist()
    print (comb(len(list_vars), n_features), 'combinations')
    
    t = (comb(len(list_vars), n_features)*5)/1200
    # convert t to hours
    t = t/60
    print ('Estimated time: ', round(t), 'hours')
    for combo in combinations(list_vars, n_features):
        nr += 1
        X = data[list(combo)]
        cols.append(X.columns)
        if X.shape[1] == n_features:
            scores = cross_val_score(model, X, target, cv=crossval, scoring=scoring)
            r2 = scores.mean()
            score.append(r2)
            if r2 > max_score:
                max_score = r2
                max_vars = X.columns
                max_nr = nr
                print (max_score, max_vars, max_nr)
    return (max_score, max_vars, max_nr,score,cols)



from imblearn.under_sampling import RandomUnderSampler

class StratifiedRandomUnderSampler():
    
    def __init__(self,df,X_name,y_name,groupvar_name,sampling_strategy='auto',
                 return_indices=False,random_state=None,replacement=False,ratio=None):
        
        self.df = df
        self.X_name = X_name
        self.y_name = y_name
        self.groupvar_name = groupvar_name
        self.sampling_strategy = sampling_strategy
        self.return_indices = return_indices
        self.random_state = random_state
        self.replacement = replacement
        self.ratio = ratio
        
    def resample(self):
        self.df_res_ = self.df.groupby(self.groupvar_name).apply(self._random_undersample,
                                  X_name=self.X_name,
                                  y_name=self.y_name
                                  )
        
        self.df_res_.set_index(self.df_res_.index.get_level_values(1),inplace=True)
        
        self.X_ = self.df_res_[self.X_name]
        self.y_ = self.df_res_[self.y_name]
        
        return self.X_, self.y_, self.df_res_
        
    def _random_undersample(self,group_df,X_name,y_name):
        # resampling is only possible when there is more than one class label
        # NOTE: this means that groups which contain samples with only one class
        # label will be dropped.
        if group_df[y_name].nunique() > 1:
            
            # convert feature column into (n_samples,1) dimensional numpy array
            # NOTE: this reshaping is required by imblearn when dealing with one
            # dimensional feature matrices.
            X = group_df[X_name].values.reshape(-1,1)
            
            rus = RandomUnderSampler(sampling_strategy=self.sampling_strategy,
                                     return_indices=self.return_indices,
                                     random_state=self.random_state,
                                     replacement=self.replacement,
                                     ratio=self.ratio)
    
            rus.fit_resample(X,group_df[y_name])
            
            indices = rus.sample_indices_
            
            return group_df.iloc[indices]
