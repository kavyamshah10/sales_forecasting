import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df['Date']=pd.to_datetime(df['Date'])
    df=df.sort_values(by='Date')

    df.fillna(method='ffill',inplace=True)
    return df

def save_data(df,path):
    df.to_csv(path,index=False)

if __name__=="__main__":
    train=load_data("data/raw/train_data.csv")
    test=load_data("data/raw/test_data.csv")

    train=preprocess(train)
    test=preprocess(test)

    save_data(train,"data/processed/train.csv")
    save_data(test,"data/processed/test.csv")

    print("data preprocessing completed")