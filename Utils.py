from typing import Tuple


def bestFeatures (data, target,crossval,model,n_features=5,scoring='r2') -> Tuple[int, list, int, list,list]:
    from itertools import combinations
    from sklearn.model_selection import cross_val_score
    from math import comb
    """Tool to test all of possible combinations features list_vars
    WARNING: THIS IS A VERY SLOW PROCESS, USE IT ONLY FOR SMALL DATASETS
    ABOUT 5 MINUTES FOR 1200 COMBINATIONS IN A RECENT CPU
    
    By Helder Fraga helderfraga@gmail.com
    
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