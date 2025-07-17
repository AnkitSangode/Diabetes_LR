import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok = True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(x_train,x_test,y_train,y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(x_train,y_train)

            y_train_pred = model.predict(x_train)

            y_test_pred = model.predict(x_test)

            train_accuracy = accuracy_score(y_train,y_train_pred)

            test_accuracy = accuracy_score(y_test,y_test_pred)

            # Precision =  precision_score(y_test, y_test_pred)
            # Recall = recall_score(y_test, y_test_pred)
            # F1_Score =  f1_score(y_test, y_test_pred)
            # Confusion_Matrix =  confusion_matrix(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_accuracy

        return report
    except Exception as e:
        raise CustomException(e,sys)