# This file is the actual code for the custom Python algorithm linear-svc_1
from dataiku.doctor.plugins.custom_prediction_algorithm import BaseCustomPredictionAlgorithm
#from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import LinearSVC¶

class CustomPredictionAlgorithm(BaseCustomPredictionAlgorithm):    
    """
        Class defining the behaviour of `linear-svc_1` algorithm:
        - how it handles parameters passed to it
        - how the estimator works

        Example here defines an Adaboost Regressor from Scikit Learn that would work for regression
        (see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)

        You need to at least define a `get_clf` method that must return a scikit-learn compatible model

        Args:
            prediction_type (str): type of prediction for which the algorithm is used. Is relevant when 
                                   algorithm works for more than one type of prediction.
                                   Possible values are: "BINARY_CLASSIFICATION", "MULTICLASS", "REGRESSION"
            params (dict): dictionary of params set by the user in the UI.
    """
    
    def __init__(self, prediction_type=None, params=None):        
        self.clf = LinearSVC¶( penalty='l2',
                              loss='squared_hinge',
                              dual=True,  
                              C=1.0, 
                              multi_class='ovr', 
                              fit_intercept=True,
                              intercept_scaling=1, 
                              class_weight=None, 
                              verbose=0, 
                              random_state=None, 
                              max_iter=1000)
        super(CustomPredictionAlgorithm, self).__init__(prediction_type, params)
    
    def get_clf(self):
        """
        This method must return a scikit-learn compatible model, ie:
        - have a fit(X,y) and predict(X) methods. If sample weights
          are enabled for this algorithm (in algo.json), the fit method
          must have instead the signature fit(X, y, sample_weight=None)
        - have a get_params() and set_params(**params) methods
        """
        return self.clf