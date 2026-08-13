# Transformer Model for Stock Price Prediction

A Transformer built **from scratch**, step by step, to investigate whether news and macroeconomic context can improve next-day stock return prediction for a specific stock.

The core idea: use the attention mechanism as a **context-retrieval system** — given the current market environment, the model identifies similar past contexts and uses them to inform its prediction, rather than treating each day as independent.

## Motivation

The same macroeconomic event can lead to very different market reactions depending on context. A rise in inflation, for example, is read very differently during periods of low vs. high interest rates, low vs. high unemployment, or in a bullish vs. bearish market. This project treats **context as a first-class input**, not an afterthought, using three data modalities:

- **Financial data** — technical indicators and stock-specific features
- **Macroeconomic data** — Fed-relevant indicators, treasury dynamics, employment and inflation data, with countdowns to publication dates (accounting for the fact that markets react to the *release* date, not the reference period)
- **News embeddings** — daily summaries of stock-specific news

## Repository structure

| File | Description |
|---|---|
| `Text_MLProject.py` | Retrieves embedded news for a given stock; supports both raw embeddings and PCA-reduced representations; includes `checkTheSimilarityEmb` to retrieve similar past text given a query |
| `MacroData_MLProject.py` | Builds macroeconomic features, correctly aligned to **publication date** rather than reference period, plus a countdown-to-release feature |
| `Stock_MLProject.py` | Builds stock-specific financial features |
| `Utilities.py` | `download_data` and `normalize` — handle dataset assembly, train/test split, and normalization |
| `TransformerForTimeSeriesStepByStepFixed.ipynb` | Full from-scratch implementation of the Transformer, built step by step |

## Usage

```python
# Download the data
merge_df, TEXT_FEATURES_ALL, FIN_FEATURES_ALL, MACRO_FEATURES_ALL, \
macro_features_with_days_to_next, all_cols_with_lags = download_data(
    "TSLA",                 # Ticker
    "Elon Musk",            # CEO of the company
    "Tesla",                # Company name
    "PCA",                  # or "embedding" to use full embeddings
    30,                     # Number of PCA components
    0                       # Number of lags
)

merge_df["Ticker"] = 0
merge_df.dropna(inplace=True)

# Normalize and split
train_df, test_df, target_scaler = normalize(
    merge_df,
    "Return",   # Target column (e.g. "Volatility_5")
    -1,         # Shift (e.g. -5 for Volatility_5)
    MACRO_FEATURES_ALL,
    FIN_FEATURES_ALL,
    TEXT_FEATURES_ALL,
    macro_features_with_days_to_next
)
```

The classes are designed to generalize across different tickers and targets — swap the ticker/CEO/company name and target column to run the pipeline on any S&P 500 stock.

## Tech stack

Python · PyTorch · Transformers (implemented from scratch, no pre-built architecture)