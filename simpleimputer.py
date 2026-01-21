from sklearn.impute import SimpleImputer
import numpy as np

data = [[90],[np.nan],[70],[80],[np.nan],[np.nan],[10],[20]]

imputer = SimpleImputer(strategy='constant',fill_value=0)
new = imputer.fit_transform(data)
print(new)