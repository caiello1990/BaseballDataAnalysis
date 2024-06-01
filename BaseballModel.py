import pandas as pd

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
df = pd.read_csv('df_bp1.csv')

# df.info(max_cols=1000)

df.season.value_counts().sort_index()

hv_mean = df.home_victory.mean()
df.run_diff.value_counts()

df.outs_total.value_counts()

df.home_victory[df.outs_total == 53].mean()
# df.loc[(df.outs_total == 53) & (df.home_victory != 1), :]
df.game_no_h.value_counts().sort_index()
