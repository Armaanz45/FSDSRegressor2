import numpy as np 
import pandas as pd 
import os 
import sys 
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import pickle

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path) #path of the pickle file & preprocessor object 

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

    ##Overall, this function takes an object (obj) and a file path (file_path) as input, creates any necessary directories, 
    # opens the file in write byte mode, serializes the object using pickle, 
    # and writes the serialized data to the file.

    ##This line extracts the directory path from the file_path argument. 
    # The os.path.dirname() function is used to extract the directory path from a file path.


def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)

               