import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def split_data(df,split_year=2012):
    df['Date']=pd.to_datetime(df['Date'],format='%d-%m-%Y')
    df['Year']=df['Date'].dt.year

    train_df=df[df['Year']<split_year]
    test_df=df[df['Year']==split_year]

    return train_df,test_df

def save_data(train_df,test_df):
    train_df.to_csv("data/raw/train_data.csv",index=False)
    test_df.to_csv("data/raw/test_data.csv",index=False)

if __name__=="__main__":
    df=pd.read_csv("data/raw/Walmart_Sales.csv")

    train_df,test_df=split_data(df)
    save_data(train_df,test_df)

    print("Data spliting completed")