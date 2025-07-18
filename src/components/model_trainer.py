import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('splitting train and test input data')
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
            "LogisticRegression": LogisticRegression(),
            "AdaBoostClassifier": AdaBoostClassifier(),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "RandomForestClassifier": RandomForestClassifier()
            }

            param_grids = {
    'LogisticRegression': [
        {
            'penalty': ['l1'],
            'solver': ['liblinear', 'saga'],
            'C': [0.01, 0.1, 1, 10],
            'max_iter': [200, 500]
        },
        {
            'penalty': ['l2'],
            'solver': ['lbfgs', 'newton-cg', 'sag'],
            'C': [0.01, 0.1, 1, 10],
            'max_iter': [200, 500]
        },
        {
            'penalty': ['elasticnet'],
            'solver': ['saga'],
            'C': [0.01, 0.1, 1, 10],
            'l1_ratio': [0.1, 0.5, 0.9],
            'max_iter': [200, 500]
        }
    ],
    'AdaBoostClassifier': {
        'n_estimators': [50, 100, 200, 500],
        'learning_rate': [0.01, 0.1, 1, 10]
    },
    'KNeighborsClassifier': {
        'n_neighbors': list(range(1, 31, 2)),
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    },
    'DecisionTreeClassifier': {
        'criterion': ['gini', 'entropy'],
        'max_depth': list(range(2, 21)),
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2']
    },
    'RandomForestClassifier': {
        'n_estimators': [100, 200, 400, 800],
        'criterion': ['gini', 'entropy'],
        # 'max_depth': [None, 5, 10, 20],
        # 'min_samples_split': [2, 5, 10],
        # 'min_samples_leaf': [1, 2, 4],
        'max_features': [ 'sqrt', 'log2', None],
        # 'bootstrap': [True, False]
    }
}


            model_report :dict = evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params = param_grids)

            ## To get best model accuracy from dict
            best_model_accuracy = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_accuracy)
            ]

            best_model = models[best_model_name]

            if best_model_accuracy < [0.6]:
                raise CustomException('No model found with accuracy over 60%')
            
            logging.info('Best found model on both training and testing dataset')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            predicted = best_model.predict(x_test)
            accuracy = accuracy_score(y_test,predicted)
            return accuracy
        
        except Exception as e: 
            raise CustomException(e,sys)