import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
import re
import yfinance as yf
"""
Functions defined:
- download_text:
    first the function download the csv file containing all the news and their summary, then it filter the news based on the presence of the company name, ticker or head name in the text
    and finally it group the news by date concatenating the text of the news of the same date in a single string.
- compute_embeddings:
    this function takes as input the dataset from the prevoius function and then compute the embedding of the text using sentence transformer.
    The embedding has dimension 384, each column now represent an embedding, e.g. column embeddiing_0 is a column and so on.
    The final dataset has, date, economics_finance and embedding from 0 to 383.
- embedding_or_pca:
    as we may want the entire embedding or a reduced version of it, this function helps to choose between the two.
    From the previous function we obtained a dataset with 384 embedding. Now, if we choose "embedding" the dimension remains the same, no further operation is done.
    While, if we choose "pca", the function apply PCA to reduce the dimension of the embedding to n_component, which is a parameter of the function.

"""

class NewsEmbeddingPipeline:

    def __init__(self,choice, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.choice=choice

    def download_text(self,file_path,company_name,head_name,ticker_name):

        text=pd.read_csv(file_path)
        text=text.drop(columns=["Unnamed: 0"])
        df_text=text[text["text"].str.contains(
                f"{company_name}|{ticker_name}|{head_name}", regex=True, na=False
            )]
        df_text=df_text.drop(columns=["text"])
        df_text = df_text.groupby("Date")["economics_finance"].apply(lambda x: "\n".join(x.astype(str))).reset_index()

        return df_text


    def compute_embeddings(self,df_text):

        texts = df_text["economics_finance"].astype(str).tolist()

        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        emb_df = pd.DataFrame(
            embeddings,
            columns=[f"embedding_{i}" for i in range(embeddings.shape[1])]
        )

        df_text = pd.concat([df_text.reset_index(drop=True), emb_df], axis=1)

        df_text["Date"] = pd.to_datetime(df_text["Date"], utc=True, errors="coerce")
        df_text["Date"]=df_text["Date"].dt.date

        return df_text #It has columns: Date, economics_finance, embedding_0, embedding_1, ..., embedding_383


    def embedding_or_pca(self,n_component,df_text_with_embedding):

        if self.choice=="embedding":
            return df_text_with_embedding # So, our df with date, economics_finance and embedding_0, embedding_1, ..., embedding_383

        else: #If no choiche were made or any other string
            embedding_cols = [c for c in df_text_with_embedding.columns if c.startswith("embedding_")] # We define the embedding columns so embedding_0, embedding_1, ..., embedding_383
            self.pca = PCA(n_components=n_component) #Initialization of the pca with n_component, which is the number of component we want to keep after the reduction
            vis_dims=self.pca.fit_transform(df_text_with_embedding[embedding_cols].values) #We apply the pca to the embedding columns, we obtain a new dataset with a column containing a list of the new values with dimension n_component
            emb_df = pd.DataFrame(
                vis_dims,
                columns=[f"embedding_{i}" for i in range(n_component)]
            )
            #As we have done before, we may want to expand the new embedding in n_component columns, so we create a new dataframe with n_component columns, each column represent a component of the new embedding
            #emb_df has only the new pca columns, so from 0 to n_component-1

            df_text_with_embedding = df_text_with_embedding.drop(columns=embedding_cols) #First, we may want to drop the old embedding columns, so we drop the one we have defined before
            df_text_with_embedding = pd.concat(
                [df_text_with_embedding.reset_index(drop=True), emb_df], axis=1
            ) #Then we concatenate the new pca columns with the rest of the dataset, so we have a new dataset with date, economics_finance and embedding_0, embedding_1, ..., embedding_{n_component-1}

        return df_text_with_embedding


    def checkTheSimilarityEmb(self,df_text_with_embedding,query):
        #Is going to be, either our dataset with 384 embedding columns or the one with n_component embedding columns, depending on the choice we made before
        df_similarity=df_text_with_embedding.copy()
        q=self.model.encode(query).reshape(1,-1) #Our query is a string, we need to encode it to obtain the embedding, then for the pca, we need to reshape it to have 1 row and the same number of columns as the embedding columns in our dataset, so either 384 or n_component

        if self.choice!="embedding": #If we have chosen to apply pca, we need to transform the query embedding with the same pca we have fitted before, so we apply the transform method of the pca to the query embedding
            # Ensure pca is initialized if choice is not embedding
            if hasattr(self, 'pca') and self.pca is not None:
                q=self.pca.transform(q)
            else:
                # This case indicates an issue: PCA was chosen but not initialized.
                # For now, proceed without PCA transformation on query, but this might lead to incorrect similarity.
                # A proper solution would be to re-evaluate the flow or raise an error.
                pass # Or raise an error: raise ValueError("PCA not initialized for query transformation.")

        embedding_features = [c for c in df_similarity.columns if c.startswith("embedding_")] #Indipendently from the choice we made before, we need to define the embedding columns, so we define them as the columns that start with "embedding_", so either embedding_0, embedding_1, ..., embedding_383 or embedding_0, embedding_1, ..., embedding_{n_component-1}

        X_emb = df_similarity[embedding_features].values
        X_emb = np.nan_to_num(X_emb, nan=0.0) #replace NaN with 0

        norms = np.linalg.norm(X_emb, axis=1, keepdims=True, ord=2) #Apply L2 normalization to the embedding. Going back to our initial dataset, we had that each row was the news for a day. With the embedding we have each row representing the news. So the L2 normalization should be applied for each row. That way we obtain the L2 norm
        norms[norms==0] = 1e-8 # Add a small epsilon to avoid division by zero
        X_emb = X_emb / norms #Eahc embedding is divided by its L2 norm, for that news of that day, we have the normalized embedding

        df_similarity[embedding_features] = X_emb

        q_norm = np.linalg.norm(q) # Calculate scalar L2 norm of the query
        if q_norm != 0:
            q = q/q_norm
        # If q_norm is 0, q remains a zero vector
        scores = df_similarity[embedding_features].values @ q.T
        scores = scores.flatten()

        top_idx = np.argsort(scores)[::-1][:10]
        Mask=df_similarity.loc[top_idx]["Date"].unique()
        Mask_delta=Mask - pd.Timedelta(days=1)
        TSLA=yf.download(tickers="TSLA", start=df_similarity["Date"].min(), end=df_similarity["Date"].max())["Close"].reset_index()
        TSLA["Return"]=TSLA["TSLA"].pct_change()
        series=TSLA.set_index("Date")["Return"]
        series_delta=series.where(series.index.isin(Mask_delta)).dropna()
        series_no_delta=series.where(series.index.isin(Mask)).dropna()
        df_similarity_filtered=df_similarity.iloc[top_idx]["economics_finance"]

        return series_delta,series_no_delta,df_similarity_filtered


# pipeline = NewsEmbeddingPipeline(choice="embedding")

# file_path="/content/drive/MyDrive/ML_Project_Data/SummaryAndNews.csv"

# df_text_original=pipeline.download_text(
#         file_path,
#         "Tesla",
#         "Elon Musk",
#         "TSLA"
#     )

# df_emb=pipeline.compute_embeddings(df_text_original)
# df_text= pipeline.embedding_or_pca(30,df_emb)

# series_delta,series_no_delta, df_similarity_filtered=pipeline.checkTheSimilarityEmb(df_text,"Tesla's stock dropped")
# df_text.drop(columns=["economics_finance"],inplace=True)
# df_text.drop_duplicates(inplace=True)
# df_text.sort_values("Date",inplace=True)
# df_text.reset_index(inplace=True)
#     # True se ci sono duplicati
# has_duplicates = df_text["Date"].duplicated().any()
# print("Ci sono duplicati?", has_duplicates)
# print(series_delta)
# print("-------------")
# print(series_no_delta)
# print("-------------")
# print(df_similarity_filtered)
