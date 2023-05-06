"""Source code of your project"""
from utils.make_data import make_apolar_data

df_apol = make_apolar_data("data/raw/expd_apolar_molecules.CSV", all=False)

print(df_apol)
