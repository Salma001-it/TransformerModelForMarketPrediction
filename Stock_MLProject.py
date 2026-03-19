import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

"""
This class has the following functions:
compute_features: which returns the df_model with all the features and the list of features used for the model

"""
class StockSequenceBuilder:
    def __init__(self, ticker="TSLA", start="2016-01-01", end="2024-06-30"):
        self.ticker = ticker
        self.start = start
        self.end = end
        #First, we may want to look for the most correlated stock with the one selected
        Tickers=pd.read_excel("https://raw.githubusercontent.com/Salma001-it/TransformerModelForMarketPrediction/main/SP500CompanyNameTicker.xlsx") #Contains a list of 500 tickers
        Tickers=Tickers["Symbol"] #The tickers are in the "Symbol" column
        Tickers=Tickers.to_list()
        Data=yf.download(Tickers, start="2010-01-01", end="2026-01-01") #It downloads all the tickers
        Data=Data["Close"].pct_change() #Calculate the return
        years=Data.index.year.unique()
        df_corr=pd.DataFrame()
        # for y in (years):
        #   Current=Data.loc[Data.index.year==y] #Dataset containing only the data of the current year
        #   Current_Corr=Current.corr()  #It contains the matrix correlation
        #   Current_Corr=Current_Corr[self.ticker].sort_values(ascending=False).head(5) #Correlation with our ticker
        #   Current_Corr=Current_Corr.reset_index()["Ticker"].to_list() #Now we have the list of tickers correlated with our
        #   df_corr=pd.concat([df_corr,Data.loc[Data.index.year==y,Current_Corr]])
        # df_corr=df_corr.fillna(0)
        # self.df_corr=df_corr.drop(columns=self.ticker)

        #Usually futures may also give some insight on the future expectation of the market
        self.futures = [
            # Indici Futures 
            "NQ=F",   # Nasdaq
            "YM=F", #Dow Jones
            "ES=F", #SP500

            # Treasury Futures
            "ZB=F",   # U.S. Treasury Bond Futures
            "ZN=F",   # 10-Year T-Note
            "ZF=F",   # 5-Year T-Note
            "ZT=F",   # 2-Year T-Note

            # Metalli
            "GC=F",   # Gold
            "SI=F",   # Silver
            "PL=F",   # Platinum
            "HG=F",   # Copper

            # Energia
            "CL=F",   # Crude Oil
            "NG=F",   # Natural Gas

        ]

    def compute_features(self):
        start_dt, end_dt = self.start, self.end
        #Download the selected ticker stock price
        tsla = yf.download(self.ticker, start=start_dt, end=end_dt)
        #Usually they have two levels
        tsla.columns = tsla.columns.droplevel(1)

        tsla[f"Return"] = np.log(tsla["Close"] / tsla["Close"].shift(1))
        tsla[f"Return_5"] = np.log(tsla["Close"] / tsla["Close"].shift(5))

        # # Medie Mobili
        tsla['SMA_20'] = tsla['Close'].rolling(window=20).mean()
        tsla['SMA_50'] = tsla['Close'].rolling(window=50).mean()
        tsla['EMA_20'] = tsla['Close'].ewm(span=20, adjust=False).mean()
        tsla['EMA_50'] = tsla['Close'].ewm(span=50, adjust=False).mean()

        # RSI
        delta = tsla['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        tsla['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = tsla['Close'].ewm(span=12, adjust=False).mean()
        exp2 = tsla['Close'].ewm(span=26, adjust=False).mean()
        tsla['MACD'] = exp1 - exp2
        tsla['Signal_Line'] = tsla['MACD'].ewm(span=9, adjust=False).mean()

        tsla['Intra_Day_Mov'] = tsla['High'] - tsla['Low']

        # # Indicatori Booleani
        # tsla['RSI_overbought'] = (tsla['RSI'] > 70).astype(int)
        # tsla['RSI_oversold'] = (tsla['RSI'] < 30).astype(int)
        # tsla['MACD_cross_up'] = (tsla['MACD'] > tsla['Signal_Line']).astype(int)

        #Main indexes returns
        market = yf.download(["^GSPC","^IXIC","^DJI"], start=start_dt, end=end_dt)["Close"]
        market["Return_SP500"] = np.log(market["^GSPC"] / market["^GSPC"].shift(1))
        market["Return_Nasdaq"] = np.log(market["^IXIC"] / market["^IXIC"].shift(1))
        market["Return_DowJones"] = np.log(market["^DJI"] / market["^DJI"].shift(1))
        market = market[["Return_SP500","Return_Nasdaq","Return_DowJones"]]

        # Merge tickers returns with market returns
        df = tsla.merge(market, left_index=True, right_index=True)
        # Feature cross-asset
        df[f"Excess_vs_SP500"] = df[f"Return"] - df["Return_SP500"]
        df[f"Excess_vs_Nasdaq"] = df[f"Return"] - df["Return_Nasdaq"]

        rolling_window = 15

        df[f"Rolling_Corr_SP500"] = df[f"Return"].rolling(rolling_window).corr(df["Return_SP500"])
        cov = df[f"Return"].rolling(rolling_window).cov(df["Return_SP500"])
        var = df["Return_SP500"].rolling(rolling_window).var()
        df[f"Rolling_Beta_SP500"] = cov / var
        df["Market_Volatility"] = df[["Return_SP500","Return_Nasdaq"]].rolling(rolling_window).std().mean(axis=1)
        df[f"Volatility"] = df[f"Return"].rolling(rolling_window).std()*np.sqrt(252)
        df[f"Volatility_5"] = df[f"Return"].rolling(5).std()*np.sqrt(252)
        df["Intra_Day_Range"] = df["High"] - df["Low"]
        df["Vol_Ratio"] = df["Return"].rolling(5).std() / df["Return"].rolling(21).std()
        df["Volume_ZScore"] = ((tsla["Volume"] - tsla["Volume"].rolling(21).mean()) /tsla["Volume"].rolling(21).std())
        # As they were too many futures and many of them are correlated, I've used PCA to reduce their dimension
        fut_prices = yf.download(self.futures, start=start_dt, end=end_dt)["Close"]
        fut_returns = np.log(fut_prices / fut_prices.shift(1)).dropna()

        scaler_fut = StandardScaler()
        X_fut_scaled = scaler_fut.fit_transform(fut_returns)

        pca = PCA(n_components=0.95) #In order to mantain 95% of the original data variance
        X_fut_pca = pca.fit_transform(X_fut_scaled)
        df_fut_pca = pd.DataFrame(X_fut_pca, index=fut_returns.index,
                                  columns=[f"Fut_PC{i+1}" for i in range(X_fut_pca.shape[1])])

        # Merge the dataframe with PCA futures
        df_model = df.merge(df_fut_pca, left_index=True, right_index=True, how="inner")
        #df_model=df_model.merge(self.df_corr,left_index=True,right_index=True,how="inner")
        df_model["Ticker"]=self.ticker

        # Selezione features finali
        features = [
            "Return",
            "Volatility",
            "SMA_20",
            "SMA_50",
            "EMA_20",
            "EMA_50",
            "RSI",
            "MACD",
            "Signal_Line",
            "Intra_Day_Mov",
            "Excess_vs_SP500",
            "Excess_vs_Nasdaq",
            "Rolling_Corr_SP500",
            "Rolling_Beta_SP500",
            "Market_Volatility",
            "Return_5",
            "Ticker",
            "Vol_Ratio",
            "Volume_ZScore",
            "Volatility_5"

        ] + list(df_fut_pca.columns) #+ self.df_corr.columns.to_list()

        df_model = df_model[features]

        #Final clean of the data
        df_model.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_model = df_model.dropna()
        return df_model,features #This function return the df_model and its features

 
# if __name__ == "__main__":
#     builder = StockSequenceBuilder(ticker="TSLA", start="2010-01-01", end="2024-06-30",
#                                    window=10, top_k=5)
#     df_mode, features=builder.compute_features()