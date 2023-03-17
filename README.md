# Utils
#Utilities to help with data science in python

# bestFeatures function:

""" bestFeatures function:

    Tool to test all of possible combinations features list_vars
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
    max_score, max_vars, max_nr,score,cols = bestFeatures (data, target, crossval, model, 10)
    """
