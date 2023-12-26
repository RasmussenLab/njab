# %%
from pathlib import Path
import pandas as pd

# %%
fname_curated_data = 'Alzheimer.xlsx'

FOLDER = Path('alzheimer').absolute()
FOLDER.mkdir(exist_ok=True, parents=True)
files_out = dict()

# %%
data = pd.read_excel(fname_curated_data)
data.index = [f'Sample_{i:03d}' for i in data.index]
data.index.name = 'Sample ID'
data.head()

# %%
meta = data.filter(like='_', axis=1)
meta.sample(5)

# %% [markdown]
# # Meta data dump
# - N=197 samples with complete data for
#   - age
#   - gender
#   - primary biochemical Alzheimer disease classification

# %%
cols = [
    '_age at CSF collection',
    '_gender',
    '_primary biochemical AD classification'
]
meta[cols].dropna().describe(include='all')


# %%
y_obj = meta['_primary biochemical AD classification'].rename('AD')
y = pd.get_dummies(y_obj)["biochemical AD"].rename('AD').to_frame()
y

# %% [markdown]
# Encode some clincial variables as binary

# %% tags=["hide-input"]
# ! preprocessing should move to setup script
dummies_collection_site = pd.get_dummies(
    meta['_collection site'])  # reference is first column (Berlin)
dummies_collection_site.describe()

# %% tags=["hide-input"]
clinic_for_ml = dummies_collection_site.iloc[:, 1:]
clinic_for_ml = (clinic_for_ml.join(
    pd.get_dummies(meta['_gender']).rename(columns={
        'f': 'female',
        'm': 'male'
    }).iloc[:, 1:])).assign(age=meta['_age at CSF collection']).join(
        y
)
# clinic_for_ml = clinic[["EGFR", "M"]]
clinic_for_ml

# %%
fname = FOLDER / 'clinic_ml.csv'
files_out[fname.stem] = fname
clinic_for_ml.to_csv(fname)

# %%
fname = FOLDER / 'meta.csv'
files_out[fname.stem] = fname
meta.to_csv(fname)

# %% [markdown]
# # Proteome dump
# %%
data = data.iloc[:, :-11]
data.sample(5)

# %%
ax = data.notna().sum(axis=0).plot.box()

# %%
fname = FOLDER / 'proteome.csv'
files_out[fname.stem] = fname
data.to_csv(fname)

# %% [markdown]
# Protein Group - Gene Mapping

# %%

protein_groups = data.columns.to_list()
proteins_unique = set()
for pg in protein_groups:
    proteins_unique |= set(pg.split(';'))
# proteins_unique

# %%
files_out

# %%
