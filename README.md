# (not) Just Another Biomarker (nJAB)

`njab` is a collection of some python function building on top of 
`pandas`, `scikit-learn`, `statsmodels`, `pingoin`, `numpy` and more...

It aims to formalize a procedure for biomarker discovery which was first developed for 
a paper on alcohol-related liver disease, based on mass spectrometry-based proteomics
measurements of blood plasma samples:

> Niu, L., Thiele, M., Geyer, P. E., Rasmussen, D. N., Webel, H. E.,  
> Santos, A., Gupta, R., Meier, F., Strauss, M., Kjaergaard, M., Lindvig,  
> K., Jacobsen, S., Rasmussen, S., Hansen, T., Krag, A., & Mann, M. (2022).  
> “Noninvasive Proteomic Biomarkers for Alcohol-Related Liver Disease.”  
> Nature Medicine 28 (6): 1277–87.  
> [nature.com/articles/s41591-022-01850-y](https://www.nature.com/articles/s41591-022-01850-y)

The approach was formalized for an analysis of inflammation markers of a cohort of patients with alcohol related cirrhosis, 
based on OLink-based proteomics measurments of blood plasma samples:
> Mynster Kronborg, T., Webel, H., O’Connell, M. B., Danielsen, K. V., Hobolth, L., Møller, S., Jensen, R. T., Bendtsen, F., Hansen, T., Rasmussen, S., Juel, H. B., & Kimer, N. (2023).  
> Markers of inflammation predict survival in newly diagnosed cirrhosis: a prospective registry study.  
> Scientific Reports, 13(1), 1–11.  
> [nature.com/articles/s41598-023-47384-2](https://www.nature.com/articles/s41598-023-47384-2)

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
