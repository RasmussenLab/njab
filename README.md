# (not) Just Another Biomarker (nJAB)

`njab` is a collection of some python function building on top of 
`pandas`, `scikit-learn`, `statsmodels`, `pingoin`, `numpy` and more...

It aims to provide a procedure for biomarker discovery which was first developed for 
a paper on fatty liver disease. 

## Installation

Install using pip from [PyPi](https://pypi.org/project/njab) version.

```
pip install njab
```

or directly from github

```
pip install git+https://github.com/RasmussenLab/njab.git
```

## Tutorials

The tutorial can be found on the documentation of the project with output
or can be run directly in colab.

### Explorative Analysis of survival dataset

[![open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RasmussenLab/njab/blob/HEAD/docs/tutorial/explorative_analysis.ipynb)

The tutorial builds on a dataset example of survival of prostatic cancer.

The main steps in the tutorial are:

1. Data loading and inspection
2. Uncontrolled binary and t-tests for binary and continous variables respectively
3. ANCOVA analysis controlling for age and weight, corrected for multiple testing
4. Kaplan-Meier plots of for significant features

### Biomarker discovery tutrial 

[![open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RasmussenLab/njab/blob/HEAD/docs/tutorial/log_reg.ipynb)


All steps are describe in the tutorial, where you could load your own data with minor adaptions.
The tutorial build on an curated [Alzheimer dataset from omiclearn](https://github.com/MannLabs/OmicLearn/tree/master/omiclearn/data). See the [Alzheimer Data](https://github.com/RasmussenLab/njab/tree/HEAD/docs/tutorial/data) section for more information.

The main steps in the tutorial are:

1. Load and prepare data for machine learning
2. Find a good set of features using cross validation
3. Evaluate and inspect your model retrained on the entire training data

## Documentation

Please find the documentation under [njab.readthedocs.io](https://njab.readthedocs.io)
