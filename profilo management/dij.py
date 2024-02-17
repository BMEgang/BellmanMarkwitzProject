from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
import pandas as pd

TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2015-01-01'
TRADE_START_DATE = '2016-01-01'
TRADE_END_DATE = '2018-09-01'
draw_folder = "/Users/ganghu/Desktop/pythonProject1/draw"

df_dji = YahooDownloader(
    start_date=TRADE_START_DATE, end_date=TRADE_END_DATE, ticker_list=["dji"]
).fetch_data()
df_dji = df_dji[["date", "close"]]
fst_day = df_dji["close"][0]
dji = pd.merge(
    df_dji["date"],
    df_dji["close"].div(fst_day).mul(10000),
    how="outer",
    left_index=True,
    right_index=True,
).set_index("date")

dji.to_csv(f"{draw_folder}/dji.csv")