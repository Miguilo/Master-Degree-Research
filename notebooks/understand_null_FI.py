import pandas as pd
from sklearn.neural_network import MLPRegressor
import pickle
import sys
sys.path.append("../src/")
from utils.evaluation import FI_shap_values

df_apol = pd.read_csv("../data/processed/processed_apolar.csv")

# F.I para $\pi$ e $\alpha$
x = df_apol[['Alpha', 'pi_bond']].values
y = df_apol["Expt"].values

all_models = pickle.load(open("../models/apolar/pi_alpha/all_molecules_models.sav", "rb"))
nn = all_models[3]

FI_shap_values(nn, x, y)