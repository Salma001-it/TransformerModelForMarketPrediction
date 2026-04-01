# Transformer model for stock price prediction
The goal of this project is to investigate whether news and macroeconomic context can improve next-day stock return prediction for a specific stock.
The main idea is to use the attention mechanism of a Transformer as a context retrieval system: given the current market environment, the model should identify similar past context and use It for the prediction.
## Example
Suppose we observe an increase in the inflation rate. The market reaction depends strongly on the surrounding context. The same increase in inflation can occur in very different environments: during periods of low or high interest rates, low or high unemployment, or in a bullish versus a bearish market. In each of these situations, the market may react in a completely different way to the same event.

Here, I want to emphasize the importance of the environment in which a specific event occurs. The context in which the event happens plays a crucial role in determining how the market reacts.
To represent the context, the dataset is composed of three modalities:

•	Financial data: technical indicators and other financial features for the chosen stock;

•	Macroeconomic data: Fed-relevant indicators, treasury dynamics, employment and inflation data with countdowns to publication dates;

•	News embeddings: daily summaries of stock-specific news;
# Repository structure
For each of the three modalities introduced before, I created a class.
So we have:

- Text_MLProject.py: contains the code to retrive news for a specific stock, already embedded, you can also chose whether to use the embedding or PCA. It also has "checkTheSimilarityEmb" function which allow to retrive a similar past text given a query.
- MacroData_MLProject.py: the main issue with building the macro features were the date of the official datasets. Usually date of the series is not the date of the release. Near the date pubblication there might be some market movement related to it. So the date of the actual pubblication is important. Also it better represent the reality: as what is known is at the pubblication date not the month before. It also build a countdown feature until the pubblication date.
- Stock_MLProject.py: it builds all the specific stock features

I've also added Utilities.py with "download_data" and "normalize" functions,they take care of the split and the normalization of the dataset, in order to have a ready train and test dataset.

Finally we have the notebook "TransformerForTimeSeriesStepByStepFixed.ipynb", which contains the implementation step by step, from scratch, of the transformer model for this project.

# Usage
```{python}

# Download the data
merge_df, TEXT_FEATURES_ALL, FIN_FEATURES_ALL, MACRO_FEATURES_ALL, \
macro_features_with_days_to_next, all_cols_with_lags = download_data(
    "TSLA",                 # Ticker you want to download
    "Elon Musk",            # CEO of the company
    "Tesla",                # Company name
    "PCA",                  # Write "embedding" if you want to use embeddings
    30,                     # Number of PCA components
    0                       # Number of lags (if you want to use them)
)

# The classes were designed to be generic,
# so they work with different stocks and targets as well
merge_df["Ticker"] = 0

# Remove rows with missing values
merge_df.dropna(inplace=True)

# Normalize the dataset
train_df, test_df, target_scaler = normalize(
    merge_df,
    "Return",                          # Target column, you can change it. E.g Volatility_5
    -1,                                # Number of shifts. E.g for Volatility_5 use -5
    MACRO_FEATURES_ALL,
    FIN_FEATURES_ALL,
    TEXT_FEATURES_ALL,
    macro_features_with_days_to_next
)

```
