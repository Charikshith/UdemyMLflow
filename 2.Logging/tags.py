## Creating experiment name

import warnings
import argparse
import pandas as pd
import numpy as np
import logging
from   pathlib import Path
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
    Path("./data/").mkdir(parents=True, exist_ok=True)
    data.to_csv("./data/red-wine-quality.csv", index=False)
    # Spliting to train and test
    train, test = train_test_split(data)
    train.to_csv("./data/train.csv")
    test.to_csv("./data/test.csv")
    # Assign X and Y
    train_x = train.drop(['quality'],axis=1)
    test_x = test.drop(['quality'],axis=1)
    train_y = train[['quality']]
    test_y = test[['quality']]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    # Set tracking uri for mlruns, when setting his path the runs will be stored in the specific path
    mlflow.set_tracking_uri(uri="")

    print("The tracking uri is : ", mlflow.get_tracking_uri())

    #initialize set experiment with experiment name
    
       #initialize set experiment with experiment name
    get_exp = mlflow.set_experiment(experiment_name="experiment_run")

    #get_exp = mlflow.get_experiment(exp) this only works with create experiment

    # print("Name: {}".format(get_exp.name)) # exp_create_exp_artifact
    # print("Experiment_id: {}".format(get_exp.experiment_id)) # 612685774155093960
    # print("Artifact Location: {}".format(get_exp.artifact_location)) # ../examples/2.Logging/myartifacts
    # print("Tags: {}".format(get_exp.tags)) # { 'version': 'v1'}
    # print("Lifecycle_stage: {}".format(get_exp.lifecycle_stage)) # active
    # print("Creation timestamp: {}".format(get_exp.creation_time)) # 1720798498405


    #Start the run with the experiment id
    mlflow.start_run(experiment_id=get_exp.experiment_id)
    
    tags = {
        "engineering": "ML platform",
        "release.candidate":"RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state= 3)
    lr.fit(train_x,train_y)

    pred1 = lr.predict(test_x)

    (rmse,mae,r2)= eval_metrics(test_y,pred1)
    print(" ElasticNet Model (alpha={:f}, l1_ratio={:f})".format(alpha,l1_ratio))
    print(f"RMSE:{rmse} \n MAE:{mae} \n R2:{r2}",)
    
    #l# Log parameters
    params = {
        "alpha": 0.9,
        "l1_ratio": 0.9
    }
    mlflow.log_params(params)
    # Log metrics
    metrics = {
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    }
    mlflow.log_metrics(metrics)

    # # logging the parameters
    # mlflow.log_param("alpha",alpha)
    # mlflow.log_param("l1_ratio",l1_ratio)
    # mlflow.log_metric("rmse",rmse)
    # mlflow.log_metric("mae",mea)
    # mlflow.log_metric("r2",r2)
    mlflow.sklearn.log_model(lr,"my_model1")

    mlflow.log_artifacts("data/")
    artifacts_uri= mlflow.get_artifact_uri()
    print("The artifact path is ",artifacts_uri)
    # artifacts_uri = mlflow.get_artifact_uri(artifact_path="data/train.csv")
    # print("The artifact path is",artifacts_uri)

    mlflow.end_run()