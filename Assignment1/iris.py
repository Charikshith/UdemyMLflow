## Creating experiment name

import warnings
import argparse
import pandas as pd
import numpy as np
import logging
from   pathlib import Path
import mlflow
import mlflow.sklearn  
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# m1=LogisticRegression()
# m2=KNeighborsClassifier()
# m3=RandomForestClassifier()

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#get the arguments from the command shell
parser = argparse.ArgumentParser()
parser.add_argument("--n",type=float,required=False,default=100)

args = parser.parse_args()

#evaluation
def eval_metrics(actual,pred):
    accuracy = accuracy_score(actual, pred)
    conf_matrix = confusion_matrix(actual, pred)
    class_report = classification_report(actual, pred)
    return accuracy,conf_matrix,class_report

if __name__=='__main__':
    warnings.filterwarnings("ignore")
    np.random.seed(3)

    #Read the csv file from the path mentioned
    df = pd.read_csv("D:/Code/MLflow/MLops/UdemyMLflow/Assignment1/iris.csv")
    
    encoder=LabelEncoder()
    df['variety']=encoder.fit_transform(df['variety'])

    # Spliting to train and test
    train, test = train_test_split(df,test_size=0.3, random_state=42)

    # Assign X and Y
    train_x = train.drop(['variety'],axis=1)
    test_x = test.drop(['variety'],axis=1)
    train_y = train[['variety']]
    test_y = test[['variety']]




    # Set tracking uri for mlruns, when setting his path the runs will be stored in the specific path
    mlflow.set_tracking_uri(uri="")

    print("The tracking uri is : ", mlflow.get_tracking_uri())

    #initialize set experiment with experiment name
    
       #initialize set experiment with experiment name
    get_exp = mlflow.set_experiment(experiment_name="assg_iris")

    #Start the run with the experiment id
    mlflow.start_run(experiment_id=get_exp.experiment_id)
    
    tags = {
        "model": "RF",
        "data":"iris",
        "Assignment": "1.0"
    }
    n= args.n
    
    mlflow.set_tags(tags)
    lr = RandomForestClassifier(n_estimators=n, random_state= 3)
    lr.fit(train_x,train_y)

    pred1 = lr.predict(test_x)

    (accuracy,conf_matrix,class_report)= eval_metrics(test_y,pred1)
    print(" RF Model (estimators={:f})".format(n))
    print(f"accuracy:{accuracy} \n conf_matrix:{conf_matrix} \n class_report:{class_report}",)
    
    #l# Log parameters
    params = {
        "estimators": n,
            }
    mlflow.log_params(params)
    # Log metrics
    metrics = {
        "accuracy": accuracy,
        
        
    }
    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(lr,"my_rfmodel1")

    mlflow.log_artifacts("data/")
    artifacts_uri= mlflow.get_artifact_uri()
    print("The artifact path is ",artifacts_uri)
    # artifacts_uri = mlflow.get_artifact_uri(artifact_path="data/train.csv")
    # print("The artifact path is",artifacts_uri)

    mlflow.end_run()
    run = mlflow.last_active_run()
    print("Active run id is {}".format(run.info.run_id)) # a94c9fd8a4cd4b9f912f7d0709125444
    print("Active run name is {}".format(run.info.run_name))