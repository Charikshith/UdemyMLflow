import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow, mlflow.sklearn
from pathlib import Path

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--alpha",type=float,required=False,default=0.2)
parser.add_argument("--l1_ratio",type=float,required=False,default=0.6)
args=parser.parse_args()

# Evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    data = pd.read_csv("D:/Code/MLflow/MLops/UdemyMLflow/data/red-wine-quality.csv")

    train,test= train_test_split(data,random_state=4,test_size=0.3)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio=args.l1_ratio

    mlflow.set_tracking_uri(uri="")
    print("The set tracking", mlflow.get_tracking_uri())

    exp=mlflow.set_experiment(experiment_name="mlrun")
    print("Name:{}".format(exp.experiment_id))
    print("Experiment_id:{}".format(exp.experiment_id))

    mlflow.start_run(run_name='run1')

    tag={"version":"v1", "dev":"d1","release":"v1.0"}
    mlflow.set_tags(tags=tag)

    current_run=mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    en = ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=4)
    en.fit(train_x,train_y)

    pred =en.predict(test_x)
    (rmse,mae,r2) = eval_metrics(test_y,pred)

    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

        # Log parameters
    params ={ "alpha": alpha,
        "l1_ratio": l1_ratio
    }
    mlflow.log_params(params)
    # Log metrics
    metrics = {
        "rmse":rmse,
        "r2":r2,
        "mae":mae
    }
    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(en,"modelv1")
    mlflow.end_run()
    
    mlflow.start_run(run_name='run2')

    tag={"version":"v2", "dev":"d2","release":"v2.0"}
    mlflow.set_tags(tags=tag)

    current_run=mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    en = ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=4)
    en.fit(train_x,train_y)

    pred =en.predict(test_x)
    (rmse,mae,r2) = eval_metrics(test_y,pred)

    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

        # Log parameters
    params ={ "alpha": alpha,
        "l1_ratio": l1_ratio
    }
    mlflow.log_params(params)
    # Log metrics
    metrics = {
        "rmse":rmse,
        "r2":r2,
        "mae":mae
    }
    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(en,"modelv2")
    mlflow.end_run()

    mlflow.start_run(run_name='run3')

    tag={"version":"v3", "dev":"d3","release":"v3.0"}
    mlflow.set_tags(tags=tag)

    current_run=mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    en = ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=4)
    en.fit(train_x,train_y)

    pred =en.predict(test_x)
    (rmse,mae,r2) = eval_metrics(test_y,pred)

    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

        # Log parameters
    params ={ "alpha": alpha,
        "l1_ratio": l1_ratio
    }
    mlflow.log_params(params)
    # Log metrics
    metrics = {
        "rmse":rmse,
        "r2":r2,
        "mae":mae
    }
    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(en,"modelv3")
    mlflow.end_run()