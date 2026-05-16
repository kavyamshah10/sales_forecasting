import pandas as pd

def create_features(df):
    df['Date']=pd.to_datetime(df['Date'])
    
    df=df.sort_values(by=['Store','Date'])


    df['Year']=df['Date'].dt.year
    df['Month']=df['Date'].dt.month
    df['week']=df['Date'].dt.isocalendar().week
    df['Day']=df['Date'].dt.day
    df['Day_of_week']=df['Date'].dt.dayofweek

    df['Lag_1']=df.groupby(['Store'])['Weekly_Sales'].shift(1)
    df['Lag_7']=df.groupby(['Store'])['Weekly_Sales'].shift(7)

    df['Rolling_mean_7']=df.groupby(['Store'])['Weekly_Sales'].shift(7).rolling(7).mean()

    df=df.fillna(0)
    return df


if __name__=="__main__":
    train=pd.read_csv("data/processed/train.csv")
    test=pd.read_csv("data/processed/test.csv")

    train=create_features(train)
    test=create_features(test)

    train.to_csv("data/processed/train_fe.csv",index=False)
    test.to_csv("data/processed/test_fe.csv",index=False)

    print("Feature engineering completed")