from typing import Tuple


def bestFeatures (data, target,crossval,model,n_features=5,scoring='r2') -> Tuple[int, list, int, list,list]:
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
