import pandas as pd
import requests
from fredapi import Fred
import numpy as np
"""
This code contiains the following functions:
- get_clean_pit_series : 
    the main issue with building the macro features were the date of the official datasets. Usually date of the series is not the date of the release. 
    Near the date pubblication there might be some market movement related to it. So the date of the actual pubblication is important.
    Also it better represent the reality. As what is known is at the pubblication date not the month before
    The first function was created in order to allign the data with the actual reality. It do it with one series at time
- build_macro_dataset: 
    The first function works for one series, this allow to have it all the series 
- build_features:
    Anothe important aspect is to know when the series is going to be publicated. 
    This function creates a countdown until the publication date, and calculate a further transformation if it is needed. 
- build_features:
    Use the previous function in order to have the dataset and featuers needed
"""
class FredMacroData:
    def __init__(self, api_key):
        self.api_key = api_key
        self.fred = Fred(api_key=self.api_key)
        self.series_config = {
            "unemployment": {"code": "UNRATE", "release_id": 50}, #Monthly frequency, no further processing needed
            "cpi": {"code": "CPIAUCSL", "release_id": 10}, #Monthly frequency, need to calculate the percentage change
            "fedfunds": {"code": "EFFR", "release_id": 18}, #Daily frequency, its absolute value is enough, but when it change is meaningful, so we want booth
            "m2_money": {"code": "M2SL", "release_id": 21}, #Monthly frequency, billions of dollars. The absolute value and the pct change are both meaningful
            "vix": {"code": "VIXCLS", "release_id": 200}, #Daily frequency, only the pct change is needed
            "cpi_year": {"code":"CORESTICKM159SFRBATL","release_id":10}, #Monthly frequency, only its value is needed
            "treasury_10y": {"code": "DGS10", "release_id": 18} #Daily frequency, no further elaboration are needed
        }
        self.macro_data_list = []
        self.release_dates = {}

    def get_clean_pit_series(self, name, config):
        series_id = config["code"]
        release_id = config["release_id"]
        #First we need to download the data from FRED, they usally are in the following format: date, value. The frequency usually depend on the series selected. e.g. unemployment is monthly, while vix is daily. 
        #Another aspect to keep in mind is that the date of the series is not the date of the release.
        # For example, the unemployment rate for January is usually released in the first week of February. 
        # This means that if we want to use the unemployment rate as a feature for our model, we need to align it with the date of the release, not the date of the data.
        
        data = self.fred.get_series(series_id).to_frame(name="value")
        data.index = pd.to_datetime(data.index)
        data.ffill(inplace=True) #For the shutdown
        daily_pit_series = None

        if data.resample("M").count()["value"].iloc[0] > 1: #Dummy check. I grouped the data by month, usually monthly data should have only one value per month, if there are more than one value, it means that the data is already in a daily format and we can use it as it is. Here I'm checking only the first value
            daily_pit_series = data["value"] 
            self.release_dates[name] = None
        else:
             # We need to get the release dates from FRED, the API doesn't provide it, so we need to use tha api documentation to find the right endopoint and the release_id
            url = ("https://api.stlouisfed.org/fred/release/dates"
                   f"?release_id={release_id}&api_key={self.api_key}&file_type=json")
            response = requests.get(url).json()
            #The output of the response is in the following json format:
            # {'realtime_start': '1776-07-04',
            # 'realtime_end': '9999-12-31',
            # 'order_by': 'release_date',
            # 'sort_order': 'asc',
            # 'count': 856,
            # 'offset': 0,
            # 'limit': 10000,
            # 'release_dates': [{'release_id': 50, 'date': '1955-05-06'},
            # {'release_id': 50, 'date': '1955-06-07'},
            #The data we are interested in is in the release_dates field, which is a list of dictionaries, each containing the release_id and the date of the release. We need to convert this list into a pandas dataframe and then align it with our data.

            releases = pd.DataFrame(response["release_dates"])
            releases["release_date"] = pd.to_datetime(releases["date"])
            self.release_dates[name] = releases["release_date"].sort_values().reset_index(drop=True)
            #We need to align the data with the release dates. 
            #Here the main issue is that the data is released with a lag, for example the unemployment rate for January is released in the first week of February, this means that we need to shift the data by one month to align it with the release date.
            data["ref_month"] = data.index.to_period("M")
            data["ref_month_shifted"] = data["ref_month"].shift(-1) # Align with next month's release
            releases["ref_month"] = releases["release_date"].dt.to_period("M")
            pit_df = pd.merge(
                releases[["release_date", "ref_month"]],
                data[["value", "ref_month_shifted"]],
                left_on="ref_month",
                right_on="ref_month_shifted",
                how="inner",
            )

            # Keep only the first release for each 'ref_month', it may happen that there are multiple releases for the same month, we want to keep only the first one.
            pit_df = pit_df.sort_values("release_date").drop_duplicates(subset=["ref_month"], keep="first")

            # Create a daily series
            temp_df = pit_df[["release_date", "value"]].copy()
            temp_df.set_index("release_date", inplace=True)
            daily_pit_series = temp_df["value"].asfreq("D").ffill() #fill to get daily values. We fill forward because the value of the series is the same until the next release, for example the unemployment rate for January is released in the first week of February, this means that the value of the unemployment rate for January, known in february, is the same until the next release in March, when we will have the unemployment rate for February.


        # Rename the series to its respective name )
        return daily_pit_series.rename(name)

    def build_macro_dataset(self):
        self.macro_data_list = []

        for name, config in self.series_config.items():
            s = self.get_clean_pit_series(name, config)
            self.macro_data_list.append(s)

        self.df_macro_final = pd.concat(self.macro_data_list, axis=1)
        self.df_macro_final = self.df_macro_final.dropna(how="any")
        
        return self.df_macro_final

    def build_features(self, df):
        pct_change_series=["cpi","fedfunds","m2_money","vix"]
        ft = pd.DataFrame(index=df.index)
        for c in df.columns:
            if c in pct_change_series:
                ft[c + "_pct_change"] = df[c].pct_change() 
            elif c not in ["cpi","vix"]:
                ft[c]=df[c].copy() 

            if c not in ["vix", "treasury_10y"] and self.release_dates.get(c) is not None: #because they are daily data
                releases_sorted = self.release_dates[c] #here we want the release dates
                days_to_next = []
                for date in df.index:
                    future = releases_sorted[releases_sorted > date]
                    days_to_next.append((future.iloc[0] - date).days if len(future) > 0 else 0)
                days_to_next_arr = np.array(days_to_next, dtype=float)
                days_to_next_arr[days_to_next_arr == 0] = 1e-8
                ft[f"{c}_days_to_next"] = 1 / days_to_next_arr

        return ft.dropna()

    def get_macro_features(self):
        df_macro = self.build_macro_dataset()
        macro_daily = self.build_features(df_macro)
        macro_daily.reset_index(inplace=True)
        macro_daily.rename(columns={"index": "Date"}, inplace=True)
        macro_features = macro_daily.columns.to_list()
        macro_features.remove("Date")
        print("Feature macro giornaliere:", macro_features)
        return macro_daily, macro_features


