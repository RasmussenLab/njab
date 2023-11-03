# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Explorative Analysis
# Uses the `prostate` time series dataset provided by the `SurvSet` package.
# - [SurvSet package](https://github.com/ErikinBC/SurvSet/tree/main)

# %%
# %pip install njab pyopenxl

# %%
from functools import partial
from pathlib import Path
import logging

from IPython.display import display

import numpy as np
import pandas as pd

import pingouin as pg
import sklearn
import seaborn
from lifelines.plotting import add_at_risk_counts

import matplotlib.pyplot as plt

from njab.plotting.km import compare_km_curves, log_rank_test
import njab
import njab.plotting

njab.plotting.set_font_sizes('x-small')
seaborn.set_style("whitegrid")

# %% [markdown]
# ### Set parameters

# %% tags=["parameters"]
TARGET = 'event'
TIME_KM = 'time'
FOLDER = 'prostate'
CLINIC = 'https://raw.githubusercontent.com/ErikinBC/SurvSet/main/SurvSet/_datagen/output/prostate.csv'
val_ids: str = ''  # List of comma separated values or filepath
#
# list or string of csv, eg. "var1,var2"
clinic_cont = ['age']
# list or string of csv, eg. "var1,var2"
clinic_binary = ['male', 'AD']
# List of comma separated values or filepath
da_covar = 'num_age,num_wt'

# %%
print(f"Time To Event: {TIME_KM} and rate variables for {TARGET}")

# %%
FOLDER = Path(FOLDER)
FOLDER.mkdir(exist_ok=True, parents=True)
FOLDER

# %%
clinic = pd.read_csv(CLINIC, index_col=0).dropna(how='any')
clinic.columns.name = 'feat_name'  # ! check needs to be implemented
cols_clinic = njab.pandas.get_colums_accessor(clinic)
clinic

# %%
clinic.describe(include='object')

# %%
vars_binary = ['fac_hx', 'fac_bm']
vars_binary

# %%
clinic[vars_binary] = clinic[vars_binary].astype('category')

# %%
check_isin_clinic = partial(njab.pandas.col_isin_df, df=clinic)
covar = check_isin_clinic(da_covar)
covar

# %%
vars_cont = [
    'num_age', 'num_wt', 'num_sbp', 'num_dbp', 'num_hg', 'num_sz', 'num_sg',
    'num_ap', 'num_sdate', 'fac_stage'
]
vars_cont

# %% [markdown]
# ### Collect outputs

# %%
fname = FOLDER / '1_differential_analysis.xlsx'
files_out = {fname.name: fname}
writer = pd.ExcelWriter(fname)
fname

# %% [markdown]
# ## Differences between groups defined by target

# %%
clinic

# %%
happend = clinic[TARGET].astype(bool)

# %% [markdown]
# ### Continous

# %%
var = 'num_age'
pg.ttest(clinic.loc[happend, var], clinic.loc[~happend, var])

# %%
ana_differential = njab.stats.groups_comparision.diff_analysis(
    clinic[vars_cont],
    happend,
    event_names=(TARGET, 'no event'),
)
ana_differential = ana_differential.sort_values(('ttest', 'p-val'))
ana_differential.to_excel(writer, "clinic continous", float_format='%.4f')
ana_differential

# %% [markdown]
# ### Binary

# %%
clinic[vars_binary].describe()

# %% [markdown]
# Might focus on discriminative power of
#   - DecompensatedAtDiagnosis
#   - alcohol consumption
#
# but the more accute diseases as heart disease and cancer seem to be distinctive

# %%
diff_binomial = []
for var in vars_binary:
    if len(clinic[var].cat.categories) == 2:
        diff_binomial.append(
            njab.stats.groups_comparision.binomtest(clinic[var],
                                                    happend,
                                                    event_names=(TARGET,
                                                                 'no-event')))
    else:
        logging.warning(
            f"Non-binary variable: {var} with {len(clinic[var].cat.categories)} categories"
        )

diff_binomial = pd.concat(diff_binomial).sort_values(
    ('binomial test', 'pvalue'))
diff_binomial.to_excel(writer, 'clinic binary', float_format='%.4f')
with pd.option_context('display.max_rows', len(diff_binomial)):
    display(diff_binomial)

# %%
clinic_ancova = [TARGET, *covar]
clinic_ancova = clinic[clinic_ancova].copy()
clinic_ancova.describe(include='all')

# %%
clinic_ancova = clinic_ancova.dropna(
)  # for now discard all rows with a missing feature
categorical_columns = clinic_ancova.columns[clinic_ancova.dtypes == 'category']
print("Available covariates", ", ".join(categorical_columns.to_list()))
for categorical_column in categorical_columns:
    # only works if no NA and only binary variables!
    clinic_ancova[categorical_column] = clinic_ancova[
        categorical_column].cat.codes

desc_ancova = clinic_ancova.describe()
desc_ancova.to_excel(writer, "covars", float_format='%.4f')
desc_ancova

# %%
if (desc_ancova.loc['std'] < 0.001).sum():
    non_varying = desc_ancova.loc['std'] < 0.001
    non_varying = non_varying[non_varying].index
    print("Non varying columns: ", ', '.join(non_varying))
    clinic_ancova = clinic_ancova.drop(non_varying, axis=1)
    for col in non_varying:
        covar.remove(col)

# %%
ancova = njab.stats.ancova.AncovaOnlyTarget(
    df_proteomics=clinic[vars_cont].drop(covar, axis=1),
    df_clinic=clinic_ancova,
    target=TARGET,
    covar=covar,
    value_name='')
ancova = ancova.ancova().sort_values('p-unc')
ancova = ancova.loc[:, "p-unc":]
ancova.columns = pd.MultiIndex.from_product([['ancova'], ancova.columns],
                                            names=('test', 'var'))
ancova.to_excel(writer, "olink controlled", float_format='%.4f')
ancova.head(20)

# %%
writer.close()

# %% [markdown]
# ## KM plot for top marker
# Cutoff is defined using a univariate logistic regression
#
#
# $$ \ln \frac{p}{1-p} = \beta_0 + \beta_1 \cdot x $$
# the default cutoff `p=0.5` corresponds to a feature value of:
#
# $$ x = - \frac{\beta_0}{\beta_1} $$
#
# Optional: The cutoff could be adapted to the prevalence of the target.

# %%
rejected = ancova.query("`('ancova', 'rejected')` == True")
rejected

# %%
# settings for plots
class_weight = 'balanced'
y_km = clinic[TARGET]
time_km = clinic[TIME_KM]
compare_km_curves = partial(compare_km_curves,
                            time=time_km,
                            y=y_km,
                            xlabel='Days since inflammation sample',
                            ylabel=f'rate {y_km.name}')
log_rank_test = partial(
    log_rank_test,
    time=time_km,
    y=y_km,
)
TOP_N = 2  # None = all

# %%
for marker, _ in rejected.index[:TOP_N]:  # first case done above currently
    fig, ax = plt.subplots()
    class_weight = 'balanced'
    # class_weight=None
    model = sklearn.linear_model.LogisticRegression(class_weight=class_weight)
    model = model.fit(X=clinic[marker].to_frame(), y=happend)
    print(
        f"Intercept {float(model.intercept_):5.3f}, coef.: {float(model.coef_):5.3f}"
    )
    # offset = np.log(p/(1-p)) # ! could be adapted based on proportion of target (for imbalanced data)
    offset = np.log(0.5 / (1 - 0.5))  # ! standard cutoff of probability of 0.5
    cutoff = offset - float(model.intercept_) / float(model.coef_)
    direction = '>' if model.coef_ > 0 else '<'
    print(
        f"Custom cutoff defined by Logistic regressor for {marker:>10}: {cutoff:.3f}"
    )
    pred = njab.sklearn.scoring.get_pred(model, clinic[marker].to_frame())
    ax, kmf_0, kmf_1 = compare_km_curves(pred=pred)
    res = log_rank_test(mask=pred)
    ax.set_title(
        f'KM curve for {TARGET.lower()}'
        f' and marker {marker} \n'
        f'(cutoff{direction}{cutoff:.2f}, log-rank-test p={res.p_value:.3f})')
    ax.legend([
        f"KP pred=0 (N={(~pred).sum()})", '95% CI (pred=0)',
        f"KP pred=1 (N={pred.sum()})", '95% CI (pred=1)'
    ])
    fname = FOLDER / f'KM_plot_{marker}.pdf'
    files_out[fname.name] = fname
    njab.plotting.savefig(ax.get_figure(), fname)

    # add counts
    add_at_risk_counts(kmf_0, kmf_1, ax=ax)
    fname = FOLDER / f'KM_plot_{marker}_w_counts.pdf'
    files_out[fname.name] = fname
    njab.plotting.savefig(ax.get_figure(), fname)

# %%
