import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# operation types: unary, binary, n-ary
OPERATIONS = {
    'add': lambda df, inp: df[inp[0]].add(df[inp[1]]),
    'sub': lambda df, inp: df[inp[0]].sub(df[inp[1]]),
    'mul': lambda df, inp: df[inp[0]].mul(df[inp[1]]),
    'div': lambda df, inp: df[inp[0]].div(df[inp[1]]),
    'pca': lambda df, inp: pd.DataFrame(PCA(1).fit_transform(df[inp])),
    'fillna_mean': lambda df, inp: df[inp].fillna(df[inp].mean()),
    'fillna_median': lambda df, inp: df[inp].fillna(df[inp].median()),
    'std': lambda df, inp: pd.DataFrame(StandardScaler().fit_transform(df[inp]), columns=inp),
    'minmax': lambda df, inp: pd.DataFrame(MinMaxScaler().fit_transform(df[inp]), columns=inp),
}