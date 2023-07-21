import sys
sys.path.append('../src/')

import hydra
from omegaconf import DictConfig
import pandas as pd
from utils.data import get_absolute_path
from utils.evaluation import create_graph_shap, show_metrics
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

    path_to_save_imgs_x0 = get_absolute_path(cfg.feat_importance.apolar.all)
    path_to_save_imgs_x1 = get_absolute_path(cfg.feat_importance.apolar.ei_alpha)
    path_to_save_imgs_x2 = get_absolute_path(cfg.feat_importance.apolar.pi_ei)
    path_to_save_imgs_x3 = get_absolute_path(cfg.feat_importance.apolar.pi_alpha)

    list_of_paths = [
        path_all_mol_x0,
        path_all_mol_x1,
        path_all_mol_x2,
        path_all_mol_x3
    ]
    list_of_img_paths = [
        path_to_save_imgs_x0,
        path_to_save_imgs_x1,
        path_to_save_imgs_x2,
        path_to_save_imgs_x3
    ]

    list_of_models = []

    for i in list_of_paths:
        list_of_models.append(pickle.load(open(i, 'rb')))

    for i,j in enumerate(list_of_feat_names):
        create_graph_shap(list_of_models[i], list_of_x_all[i], y_all.ravel(), j, 
                          list_of_img_paths[i], "all_molecules_FI.png")


    # For partial iso molecules
    df_apolar_partial = pd.read_csv(
        get_absolute_path(cfg.data.apolar.final.path)
    )
    x0_partial_iso = df_apolar_partial[cfg.opt.features.partial.apolar.feat1_iso].values
    x1_partial_iso = df_apolar_partial[cfg.opt.features.partial.apolar.feat2_iso].values
    x2_partial_iso = df_apolar_partial[cfg.opt.features.partial.apolar.feat3_iso].values
    x3_partial_iso = df_apolar_partial[cfg.opt.features.partial.apolar.feat4_iso].values

    list_of_feat_names = [
        ['Ei', 'Alpha', 'Pi Bond'],
        ['Ei', 'Alpha'],
        ['Ei', 'Pi Bond'],
        ['Alpha', 'Pi Bond']
    ]

    y_partial = df_apolar_partial[["Expt"]].values

    list_of_x_all = [x0_partial_iso, x1_partial_iso, x2_partial_iso, x3_partial_iso]

    path_partial_iso_mol_x0 = get_absolute_path(f"{cfg.models.apolar.all}/partial_iso_molecules_models.sav")
    path_partial_iso_mol_x1 = get_absolute_path(f"{cfg.models.apolar.ei_alpha}/partial_iso_molecules_models.sav")
    path_partial_iso_mol_x2 = get_absolute_path(f"{cfg.models.apolar.pi_ei}/partial_iso_molecules_models.sav")
    path_partial_iso_mol_x3 = get_absolute_path(f"{cfg.models.apolar.pi_alpha}/partial_iso_molecules_models.sav")

    list_of_paths = [
        path_partial_iso_mol_x0,
        path_partial_iso_mol_x1,
        path_partial_iso_mol_x2,
        path_partial_iso_mol_x3
    ]


    list_of_models = []

    for i in list_of_paths:
        list_of_models.append(pickle.load(open(i, 'rb')))

    for i,j in enumerate(list_of_feat_names):
        create_graph_shap(list_of_models[i], list_of_x_all[i], y_partial.ravel(), j, 
                          list_of_img_paths[i], "partial_iso_molecules_FI.png")

    # For partial aniso molecules.

    x0_partial_aniso = df_apolar_partial[cfg.opt.features.partial.apolar.feat1_aniso].values
    x1_partial_aniso = df_apolar_partial[cfg.opt.features.partial.apolar.feat2_aniso].values
    x2_partial_aniso = df_apolar_partial[cfg.opt.features.partial.apolar.feat3_aniso].values
    x3_partial_aniso = df_apolar_partial[cfg.opt.features.partial.apolar.feat4_aniso].values

    list_of_x_all = [x0_partial_aniso, x1_partial_aniso, x2_partial_aniso, x3_partial_aniso]

    list_of_feat_names = [
        ['Ei', 'axx', 'ayy', 'azz', 'Pi Bond'],
        ['Ei', 'axx', 'ayy', 'azz'],
        ['Ei', 'Pi Bond'],
        ['axx', 'ayy', 'azz', 'Pi Bond']
    ]
    path_partial_aniso_mol_x0 = get_absolute_path(f"{cfg.models.apolar.all}/partial_aniso_molecules_models.sav")
    path_partial_aniso_mol_x1 = get_absolute_path(f"{cfg.models.apolar.ei_alpha}/partial_aniso_molecules_models.sav")
    path_partial_aniso_mol_x2 = get_absolute_path(f"{cfg.models.apolar.pi_ei}/partial_aniso_molecules_models.sav")
    path_partial_aniso_mol_x3 = get_absolute_path(f"{cfg.models.apolar.pi_alpha}/partial_aniso_molecules_models.sav")

    list_of_paths = [
        path_partial_aniso_mol_x0,
        path_partial_aniso_mol_x1,
        path_partial_aniso_mol_x2,
        path_partial_aniso_mol_x3
    ]


    list_of_models = []

    for i in list_of_paths:
        list_of_models.append(pickle.load(open(i, 'rb')))

    for i,j in enumerate(list_of_feat_names):
        create_graph_shap(list_of_models[i], list_of_x_all[i], y_partial.ravel(), j, 
                          list_of_img_paths[i], "partial_aniso_molecules_FI.png")

if __name__ == "__main__":
    main()
