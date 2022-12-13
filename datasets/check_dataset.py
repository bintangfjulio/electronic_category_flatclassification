import pandas as pd

dataset = pd.read_csv('electronic_product_tokopedia.csv')
print(dataset['leaf'].value_counts())
print(dataset.isna().sum())