from MacroData_MLProject import FredMacroData
from Stock_MLProject import StockSequenceBuilder
from Text_MLProject import NewsEmbeddingPipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def download_data(ticker,ceo,company_name,choice,n_component,N_LAGS=14):
    #Macro
    API_KEY = API_KEY
    fred_pipeline = FredMacroData(api_key=API_KEY)
    macro_daily, macro_features = fred_pipeline.get_macro_features()
    #News
    pipeline = NewsEmbeddingPipeline(choice=choice)
    file_path="/content/drive/MyDrive/ML_Project_Data/SummaryAndNews.csv"
    df_text_original=pipeline.download_text(
            file_path,
            ticker,
            ceo,
            company_name
        )

    df_emb=pipeline.compute_embeddings(df_text_original)
    df_text= pipeline.embedding_or_pca(n_component,df_emb)
    df_text.drop(columns=["economics_finance"],inplace=True)
    df_text.drop_duplicates(inplace=True)
    df_text.sort_values("Date",inplace=True)
    df_text.reset_index(inplace=True)

    #Stock
    builder = StockSequenceBuilder(ticker=ticker, start="2010-01-01", end="2025-07-30")
    stock_data,stock_features=builder.compute_features()
    text_features=[c for c in df_text.columns if c.startswith("embedding_")]

    #Data we have:
    #stock_data->stock_features
    #macro_daily->macro_features
    #df_text->text_features
    TEXT_FEATURES = text_features #I don't think that the lag of text is good
    FIN_FEATURES = [f for f in stock_features if f != "Ticker"] #All financial features except Ticker, beacause no lag of it is needed
    MACRO_FEATURES=macro_features

    #Reset index and convert to date time for the next merge
    stock_data.reset_index(inplace=True)
    stock_data["Date"]=pd.to_datetime(stock_data["Date"])
    stock_data.sort_values("Date",inplace=True)
    df_text["Date"]=pd.to_datetime(df_text["Date"])
    df_text.sort_values("Date",inplace=True)
    macro_daily["Date"]=pd.to_datetime(macro_daily["Date"])
    macro_daily.sort_values("Date",inplace=True)

    #Merge on left in order to have all stock values, days without news are going to be set as zero
    merge_df=pd.merge(stock_data,df_text,on="Date",how="left")
    merge_df=pd.merge(merge_df,macro_daily,on="Date", how="left")

    #Filter the data
    merge_df=merge_df[merge_df["Date"]>pd.to_datetime("2010-01-01")]
    merge_df=merge_df[merge_df["Date"]<pd.to_datetime("2025-07-01")]
    merge_df.fillna(0,inplace=True) #The na values are all on the text

    #Create the lag features and update the corrispondent features list

    for col in FIN_FEATURES + MACRO_FEATURES: #TEXT_FEATURES
        for lag in range(1, N_LAGS + 1):
            merge_df[f'{col}_lag{lag}'] = merge_df[col].shift(lag)

    FIN_FEATURES_ALL = FIN_FEATURES + [
        f'{col}_lag{lag}' 
        for col in FIN_FEATURES 
        for lag in range(1, N_LAGS + 1)
    ] + ["Ticker"] #We add back the column ticker removed before

    # TEXT_FEATURES_ALL = TEXT_FEATURES + [
    #     f'{col}_lag{lag}' 
    #     for col in TEXT_FEATURES 
    #     for lag in range(1, N_LAGS + 1)
    # ]

    MACRO_FEATURES_ALL = MACRO_FEATURES + [
        f'{col}_lag{lag}' 
        for col in MACRO_FEATURES 
        for lag in range(1, N_LAGS + 1)
    ]

    TEXT_FEATURES_ALL= TEXT_FEATURES
    all_cols_with_lags = FIN_FEATURES_ALL + MACRO_FEATURES_ALL + TEXT_FEATURES_ALL

    macro_features_with_days_to_next = [feature for feature in MACRO_FEATURES_ALL if "_days_to_next" in feature]
    
    return merge_df,TEXT_FEATURES_ALL,FIN_FEATURES_ALL,MACRO_FEATURES_ALL,macro_features_with_days_to_next,all_cols_with_lags



def normalize(df,Target,shift,MACRO_FEATURES_ALL,FIN_FEATURES_ALL,TEXT_FEATURES_ALL,macro_features_with_days_to_next):
    df_norm = df.copy()
    df_norm['Date'] = pd.to_datetime(df_norm['Date'])
    df_norm = df_norm.sort_values("Date").reset_index(drop=True)

    features = MACRO_FEATURES_ALL + FIN_FEATURES_ALL + TEXT_FEATURES_ALL #At the end I've decided to normalize also the text, otherwise the values weren't balanced
    df_norm[features] = (
        df_norm[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    test_start = "2024-01-01"
    split_date = pd.to_datetime(test_start)

    train_mask = df_norm["Date"] < split_date
    test_mask  = df_norm["Date"] >= split_date


    df_norm['Target'] = df_norm[Target].shift(shift)

    # remove NaN from the shift
    df_norm = df_norm.dropna(subset=['Target'])

    scaler_macro = StandardScaler()

    features_to_normalize = [
        f for f in features
        if f not in (macro_features_with_days_to_next + ["Ticker"])
    ]

    scaler_macro.fit(
        df_norm.loc[train_mask, features_to_normalize]
    )

    df_norm[features_to_normalize] = scaler_macro.transform(
        df_norm[features_to_normalize]
    )



    target_scaler = StandardScaler()
    
    target_scaler.fit(df_norm.loc[train_mask, ['Target']])
    df_norm['Target'] = target_scaler.transform(df_norm[['Target']])


    # X_emb = df_norm[TEXT_FEATURES_ALL].values
    # X_emb = np.nan_to_num(X_emb, nan=0.0)

    # norms = np.linalg.norm(X_emb, axis=1, keepdims=True, ord=2)
    # norms[norms == 0] = 1

    # df_norm[TEXT_FEATURES_ALL] = X_emb / norms

    # X_transformer = np.hstack([
    #     df_norm[features].values, #Here we have all the features
    #     df_norm[TEXT_FEATURES_ALL].values
    # ])
    X_transformer=df_norm[features]

    print("Shape input Transformer:", X_transformer.shape)

    train_df = df_norm[train_mask].copy()
    test_df  = df_norm[test_mask].copy()
    return train_df,test_df,target_scaler
