# %%
# pip install SurvSet

# %%
from pathlib import Path

from SurvSet.data import SurvLoader

loader = SurvLoader()
# List of available datasets and meta-info
loader.df_ds.head()

# %%
df, ref = loader.load_dataset(ds_name='prostate').values()

# %%
fname = "prostate.csv"

df.to_csv(fname, index=False)
