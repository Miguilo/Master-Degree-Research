import sys
sys.path.append('../src/')

import hydra
from omegaconf import DictConfig
import pandas as pd
from utils.data import get_absolute_path
from utils.evaluation import create_graph_shap
import pickle

@hydra.main(config_path="../config", config_name="main.yaml")
def main(cfg: DictConfig):
    # For all apolar Molecules
    df_apolar_all = pd.read_csv(
        get_absolute_path(cfg.data.apolar.processed.path)
    )
    x0_all = df_apolar_all[cfg.opt.features.all.apolar.feat1].values
    x1_all = df_apolar_all[cfg.opt.features.all.apolar.feat2].values
    x2_all = df_apolar_all[cfg.opt.features.all.apolar.feat3].values
    x3_all = df_apolar_all[cfg.opt.features.all.apolar.feat4].values

    list_of_feat_names = [
        ['Ei', 'Alpha', 'Pi Bond'],
        ['Ei', 'Alpha'],
        ['Ei', 'Pi Bond'],
        ['Alpha', 'Pi Bond']
    ]

    y_all = df_apolar_all[["Expt"]].values

    list_of_x_all = [x0_all, x1_all, x2_all, x3_all]

    path_all_mol_x0 = get_absolute_path(f"{cfg.models.apolar.all}/all_molecules_models.sav")
    path_all_mol_x1 = get_absolute_path(f"{cfg.models.apolar.ei_alpha}/all_molecules_models.sav")
    path_all_mol_x2 = get_absolute_path(f"{cfg.models.apolar.pi_ei}/all_molecules_models.sav")
    path_all_mol_x3 = get_absolute_path(f"{cfg.models.apolar.pi_alpha}/all_molecules_models.sav")

    list_of_paths = [
        path_all_mol_x0,
        path_all_mol_x1,
        path_all_mol_x2,
        path_all_mol_x3
    ]
    list_of_models = []

    for i in list_of_paths:
        list_of_models.append(pickle.load(open(i, 'rb')))

    print(list_of_models[-1][0])
    # for i,j in enumerate(list_of_feat_names):
    #     create_graph_shap(list_of_models[i], list_of_x_all[i], y_all.ravel(), j)
    # create_graph_shap(list_of_models[-1], list_of_x_all[-1], y_all.ravel(), list_of_feat_names[-1])
    list_of_models[-1][0].fit(list_of_x_all[-1], y_all.ravel())
    print(list_of_models[-1][0].regressor_[0].get_feature_names_out())
    



if __name__ == "__main__":
    main()
