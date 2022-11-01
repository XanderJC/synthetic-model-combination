import pandas as pd
from pkg_resources import resource_filename

DATA_LOC = resource_filename("smc", "data_loading")

df = pd.read_csv((DATA_LOC + "/Vancomycin/MOA_rawpredictions.csv"), compression="gzip")

print(df.head())
print(df.columns)
