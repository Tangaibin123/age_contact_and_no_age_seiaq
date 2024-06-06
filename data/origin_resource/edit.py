import pandas as pd
import numpy as np

csv_path = './China_subnational_Shanghai_M_overall_contact_matrix_18.csv'
df = pd.read_csv(csv_path)


# -----save dataframe as csv----- #
csv_save_path = './data_copy.csv'
df.to_csv(csv_save_path, sep=',', index=False, header=True)

print(df)
