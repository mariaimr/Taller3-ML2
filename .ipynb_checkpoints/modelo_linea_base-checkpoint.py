#%%writefile script.py

import argparse
import joblib
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import log_loss
#Escalar Variables
from sklearn.preprocessing import MinMaxScaler


# inference functions ---------------
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "modelo_linea_base.joblib"))
    return clf


if __name__ == "__main__":

    print("extracting arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--criterion", type=str, default='gini')

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="banco_train.csv")
    parser.add_argument("--test-file", type=str, default="banco_test.csv")
    parser.add_argument(
        "--features", type=str
    )  # in this script we ask user to explicitly name features
    parser.add_argument(
        "--target", type=str
    )  # in this script we ask user to explicitly name the target

    args, _ = parser.parse_known_args()

    print("reading data")
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))
    
    print("building training and testing datasets")
    X_train = train_df[args.features.split()]
    X_test = test_df[args.features.split()]
    y_train = train_df[args.target]
    y_test = test_df[args.target]

    #MinMaxScaler
    print("MinMaxScaler")
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    # train
    print("training model")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth = None,
        criterion = args.criterion,
        n_jobs=-1,
        class_weight = 'balanced'
    )

    model.fit(X_train, y_train)
    
    #nuestras validaciones
    #print("datos: ",np.unique(X_train))
    # print accuracy
    print("validating model")
    y_1 = model.predict(X_test)
    y_pred = model.predict_proba(X_test)
    print("=========== Accuracy Score ================")
    print("Accuracy = ",metrics.accuracy_score(y_test, y_1))
    
    y_test = y_test.tolist()
    y_1 = y_1.tolist()

    print("===========Balanced Accuracy Score ================")
    print("Balanced Accuracy = ",balanced_accuracy_score(y_test, y_1))
    
   
    y_test = np.array(y_test)
    print("===========log loss ================")
    print("Log Loss = ",log_loss(y_test, y_pred))

    # persist model
    path = os.path.join(args.model_dir, "modelo_linea_base.joblib")
    joblib.dump(model, path)
    print("model persisted at " + path)
