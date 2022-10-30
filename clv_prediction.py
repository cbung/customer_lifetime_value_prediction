import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


df_ = pd.read_excel(r"online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()


# Data preproccessing
def outlier_thresholds(dataframe, variable):
    q1 = dataframe[variable].quantile(0.01)
    q3 = dataframe[variable].quantile(0.99)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def prepare_data(dataframe):
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[(dataframe["Quantity"] > 0)]
    dataframe = dataframe[(dataframe["Price"] > 0)]
    dataframe.dropna(inplace=True)
    for column in dataframe.columns:
        if (column != "Customer ID") and (dataframe[column].dtypes in ["int64", "float64", "int32", "float32"]):
            replace_with_thresholds(dataframe, column)
    return dataframe


df = prepare_data(df)


# Prepare a clv dataframe to fit into the BG/NBD Model and GAMMA GAMMA submodel
df["TotalPrice"] = df["Quantity"] * df["Price"]
analysis_date = dt.datetime(2010, 12, 11)  # two days after the last InvoiceDate.
clv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                        lambda date: (analysis_date - date.min()).days],
                                        'Invoice': lambda num: num.nunique(),
                                        'TotalPrice': lambda money: money.sum()})
clv_df.columns = clv_df.columns.droplevel(0)
clv_df.columns = ['recency_clv_weekly', 'T_weekly', 'frequency', 'monetary_clv_avg']

clv_df["monetary_clv_avg"] = clv_df["monetary_clv_avg"] / clv_df["frequency"]
clv_df = clv_df[(clv_df['frequency'] > 1)]
clv_df["recency_clv_weekly"] = clv_df["recency_clv_weekly"] / 7
clv_df["T_weekly"] = clv_df["T_weekly"] / 7


# BG/NBD Model
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(clv_df['frequency'],
        clv_df['recency_clv_weekly'],
        clv_df['T_weekly'])

# GAMMA GAMMA submodel
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(clv_df['frequency'],
        clv_df['monetary_clv_avg'])

# Find 1-3-6 months of expected number of purchases and add columns to dataframe
clv_df["expected_purc_1_months"] = bgf.conditional_expected_number_of_purchases_up_to_time(4,
                                                                                           clv_df['frequency'],
                                                                                           clv_df['recency_clv_weekly'],
                                                                                           clv_df['T_weekly'])
clv_df["expected_purc_3_months"] = bgf.conditional_expected_number_of_purchases_up_to_time(12,
                                                                                           clv_df['frequency'],
                                                                                           clv_df['recency_clv_weekly'],
                                                                                           clv_df['T_weekly'])
clv_df["expected_purc_6_months"] = bgf.conditional_expected_number_of_purchases_up_to_time(24,
                                                                                           clv_df['frequency'],
                                                                                           clv_df['recency_clv_weekly'],
                                                                                           clv_df['T_weekly'])
# Visualize the predictions
plot_period_transactions(bgf)
plt.show()

# Find expected average profit
clv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(clv_df['frequency'],
                                                                            clv_df['monetary_clv_avg'])


# Calculate 3 months CLV, using BG/NBD Model and GAMMA GAMMA submodel
clv = ggf.customer_lifetime_value(bgf,
                                  clv_df['frequency'],
                                  clv_df['recency_clv_weekly'],
                                  clv_df['T_weekly'],
                                  clv_df['monetary_clv_avg'],
                                  time=3,
                                  freq="W",
                                  discount_rate=0.01).reset_index()

# Add clv to dataframe
clv_final = clv_df.merge(clv, on="Customer ID", how="left")
clv_final["Customer ID"] = clv["Customer ID"].astype(int)


# Create segments
clv_final["segment"] = pd.qcut(clv_final["clv"], 5, labels=["low", "low-mid", "mid", "high-mid", "high"])

clv_final.drop(columns="Customer ID").groupby("segment").agg({"mean", "sum"})
