


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.ensemble import RandomForestClassifier


class RandomForestUnsupervised:    
    def __init__(self, model):
        self.Synthetic_XY = None
        self.Synthetic_Score = None
        self.Synthetic_Model = None
        self.Synthetic_X = None
        self.Synthetic_Y = None
        self.model = model
        
    # Create the synthetic data
    def make_synthetic(self, Xinput):
    
        # Synthetic data with same marginal distribution for each feature
        X = Xinput.copy()
        synthetic_X = pd.DataFrame(np.zeros((X.shape[0], X.shape[1])), columns=X.columns)
        synthetic2_X = pd.DataFrame(np.zeros((X.shape[0], X.shape[1])), columns=X.columns)
        
        nof_features = X.shape[1]
        nof_objects = X.shape[0]
        
        for f in X.columns:
            feature_values = X.loc[:, f]
            synthetic_X.loc[:, f] = np.random.choice(feature_values, nof_objects)       
     
        synthetic_X["Synthetic"] = np.ones(len(synthetic_X)).astype("str")
        X["Synthetic"] = np.zeros(len(X)).astype("str")
            
        frames = [X, synthetic_X]
        Synthetic = pd.concat(frames)
              
        self.Synthetic_XY = Synthetic
        self.Synthetic_X = Synthetic.drop(['Synthetic'], axis=1)
        self.Synthetic_Y = Synthetic["Synthetic"]
    
    
    def make_forest(self, Xinput):
 
        # Create the synthetic
        self.make_synthetic(Xinput)
               
        # Train the synthetic model
        SYN = self.model.fit(self.Synthetic_X, self.Synthetic_Y)
    
        self.Synthetic_Score = SYN.oob_score_
        self.Synthetic_Model = SYN
 

# Proximity measurement
def proximityMatrix(model, X, normalize=True):      

    terminals = model.apply(X)
    nTrees = terminals.shape[1]

    a = terminals[:,0]
    proxMat = 1*np.equal.outer(a, a)

    for i in range(1, nTrees):
        a = terminals[:,i]
        proxMat += 1*np.equal.outer(a, a)

    if normalize:
        proxMat = proxMat / nTrees

    return proxMat   


