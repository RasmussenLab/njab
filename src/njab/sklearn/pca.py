import pandas as pd
import sklearn.decomposition

def run_pca(df_wide:pd.DataFrame, n_components:int=2) -> tuple[pd.DataFrame, sklearn.decomposition.PCA]:
    """Run PCA on DataFrame and return result.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame in wide format to fit features on.
    n_components : int, optional
        Number of Principal Components to fit, by default 2

    Returns
    -------
    Tuple[pd.DataFrame, PCA]
        principal compoments of DataFrame with same indices as in original DataFrame,
        and fitted PCA model of sklearn
    """
    pca = sklearn.decomposition.PCA(n_components=n_components)
    PCs = pca.fit_transform(df_wide)
    cols = [f'principal component {i+1} ({var_explained*100:.2f} %)' for i,
            var_explained in enumerate(pca.explained_variance_ratio_)]
    PCs = pd.DataFrame(PCs, index=df_wide.index, columns=cols)
    return PCs, pca