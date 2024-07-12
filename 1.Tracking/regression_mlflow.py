import warnings
import argparse
import pandas as pd
import numpy as np
import logging

import mlflow
import mlflow.sklearn  
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#get the arguments from the command shell
parser = argparse.ArgumentParser()
parser.add_argument("--alpha",type=float,required=False,default=0.2)
parser.add_argument("--l1_ratio",type=float,required=False,default=0.5)
args = parser.parse_args()

#evaluation
def eval_metrics(actual,pred):
    rmse = np.sqrt(mean_squared_error(actual,pred)) #rootmeansquareerror
    mae =mean_absolute_error(actual,pred) 
    r2 = r2_score(actual,pred)
    return rmse,mae,r2

if __name__=='__main__':
    warnings.filterwarnings("ignore")
    np.random.seed(3)

    #Read the csv file from the path mentioned
    data = pd.read_csv("D:/Code/MLflow/MLops/UdemyMLflow/data/red-wine-quality.csv")
    
    # Spliting to train and test
    train, test = train_test_split(data)

    # Assign X and Y
    train_x = train.drop(['quality'],axis=1)
    test_x = test.drop(['quality'],axis=1)
    train_y = train[['quality']]
    test_y = test[['quality']]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    #initialize set experiment with experiment name
    exp = mlflow.set_experiment(experiment_name="experiment_1")

    #Start the run with the experiment id
    with mlflow.start_run(experiment_id=exp.experiment_id):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state= 3)
        lr.fit(train_x,train_y)

        pred1 = lr.predict(test_x)

        (rmse,mea,r2)= eval_metrics(test_y,pred1)
        print(" ElasticNet Model (alpha={:f}, l1_ratio={:f})".format(alpha,l1_ratio))
        print(f"RMSE:{rmse} \n MAE:{mea} \n R2:{r2}",)
        

        # logging the parameters
        mlflow.log_param("alpha",alpha)
        mlflow.log_param("l1_ratio",l1_ratio)
        mlflow.log_metric("rmse",rmse)
        mlflow.log_metric("mae",mea)
        mlflow.log_metric("r2",r2)
        mlflow.sklearn.log_model(lr,"my_model")