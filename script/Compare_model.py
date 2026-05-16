import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from xgboost import XGBRegressor


def load_data():
    train=pd.read_csv("data/processed/train_fe.csv")
    test=pd.read_csv("data/processed/test_fe.csv")

    return train,test

def prepare_data(train,test):
    

    X_train=train.drop(columns=['Weekly_Sales','Date'],axis=1)
    y_train=train['Weekly_Sales']

    X_test=test.drop(columns=['Weekly_Sales','Date'],axis=1)
    y_test=test['Weekly_Sales']

    return X_train,X_test,y_train,y_test


def evaluate_model(model,X_train,y_train,X_test,y_test):
    model.fit(X_train,y_train)
    preds=model.predict(X_test)

    mae=mean_absolute_error(y_test,preds)
    rmse=np.sqrt(mean_squared_error(y_test,preds))

    return mae,rmse,model


def compare_models(X_train,y_train,X_test,y_test):
    models={
        "Linear_Regression":LinearRegression(),
        "Random_Forest":RandomForestRegressor(n_estimators=200,random_state=42),
        "XGBoost":XGBRegressor(n_estimator=100,learning_rate=0.1,random_state=42)
    }

    result=[]
    best_model=None
    best_rmse=float("inf")

    for name, model in models.items():
        print(f"Training....{name}...")

        mae,rmse,trained_model=evaluate_model(model,X_train,y_train,X_test,y_test)

        result.append({
            "Model":name,
            "MAE":mae,
            "RMSE":rmse
        })

        if rmse<best_rmse:
            best_rmse=rmse
            best_model=trained_model

    return result,best_model


def save_output(result,best_model):
    pd.DataFrame(result).to_csv("output/model_comparision.csv",index=False)
    joblib.dump(best_model,"model/best_model.pkl")


if __name__=="__main__":
    train,test=load_data()

    X_train,X_test,y_train,y_test=prepare_data(train,test)

    result, best_model=compare_models(X_train,y_train,X_test,y_test)
    save_output(result,best_model)
    result_df=pd.DataFrame(result)
    print("Model Comparison Tabel")

    print(result_df.to_string(index=False))
    print("model Comparision Completed")
