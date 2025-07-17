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

            model_report :dict = evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)

            ## To get best model accuracy from dict
            best_model_accuracy = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_accuracy)
            ]

            best_model = models[best_model_name]

            if best_model_accuracy < 0.6:
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