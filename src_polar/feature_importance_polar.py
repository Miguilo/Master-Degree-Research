import sys
from os import path

file_dir = path.dirname(__file__)

sys.path.insert(1, path.join(file_dir, "../src/"))
import pickle

import hydra
import pandas as pd
from omegaconf import DictConfig

from utils.data import get_absolute_path
from utils.evaluation import create_graph_shap, show_metrics


@hydra.main(
    config_path=path.join(file_dir, "../config"), config_name="main.yaml"
)
def main(cfg: DictConfig):

    # For all polar Molecules
    df_polar_all = pd.read_csv(
        get_absolute_path(cfg.data.polar.processed.path)
    )
    x0_all = df_polar_all[cfg.opt.features.all.polar.feat1].values
    x1_all = df_polar_all[cfg.opt.features.all.polar.feat2].values
    x2_all = df_polar_all[cfg.opt.features.all.polar.feat3].values
    x3_all = df_polar_all[cfg.opt.features.all.polar.feat4].values

    list_of_feat_names = [
        ["Ei", "Alpha", "Dipole", "Pi Bond"],
        ["Ei", "Alpha", "Dipole"],
        ["Alpha", "Dipole"],
        ["Alpha", "Dipole", "Pi Bond"],
    ]

    y_all = df_polar_all[["Expt"]].values

    list_of_x_all = [x0_all, x1_all, x2_all, x3_all]

    path_all_mol_x0 = get_absolute_path(
        f"{cfg.models.polar.all}/all_molecules_models.sav"
    )
    path_all_mol_x1 = get_absolute_path(
        f"{cfg.models.polar.ei_alpha_dipole}/all_molecules_models.sav"
    )
    path_all_mol_x2 = get_absolute_path(
        f"{cfg.models.polar.alpha_dipole}/all_molecules_models.sav"
    )
    path_all_mol_x3 = get_absolute_path(
        f"{cfg.models.polar.alpha_dipole_pi}/all_molecules_models.sav"
    )

    path_to_save_imgs_x0 = get_absolute_path(cfg.feat_importance.polar.all)
    path_to_save_imgs_x1 = get_absolute_path(
        cfg.feat_importance.polar.ei_alpha_dipole
    )
    path_to_save_imgs_x2 = get_absolute_path(
        cfg.feat_importance.polar.alpha_dipole
    )
    path_to_save_imgs_x3 = get_absolute_path(
        cfg.feat_importance.polar.alpha_dipole_pi
    )

    list_of_paths = [
        path_all_mol_x0,
        path_all_mol_x1,
        path_all_mol_x2,
        path_all_mol_x3,
    ]
    list_of_img_paths = [
        path_to_save_imgs_x0,
        path_to_save_imgs_x1,
        path_to_save_imgs_x2,
        path_to_save_imgs_x3,
    ]

    list_of_models = []

    for i in list_of_paths:
        list_of_models.append(pickle.load(open(i, "rb")))

    for i, j in enumerate(list_of_feat_names):
        create_graph_shap(
            list_of_models[i],
            list_of_x_all[i],
            y_all.ravel(),
            j,
            list_of_img_paths[i],
            "all_molecules_FI.png",
        )

    # For partial iso molecules
    df_polar_partial = pd.read_csv(
        get_absolute_path(cfg.data.polar.final.path)
    )
    x0_partial_iso = df_polar_partial[
        cfg.opt.features.partial.polar.feat1_iso
    ].values
    x1_partial_iso = df_polar_partial[
        cfg.opt.features.partial.polar.feat2_iso
    ].values
    x2_partial_iso = df_polar_partial[
        cfg.opt.features.partial.polar.feat3_iso
    ].values
    x3_partial_iso = df_polar_partial[
        cfg.opt.features.partial.polar.feat4_iso
    ].values

    y_partial = df_polar_partial[["Expt"]].values

    list_of_x_all = [
        x0_partial_iso,
        x1_partial_iso,
        x2_partial_iso,
        x3_partial_iso,
    ]

    path_partial_iso_mol_x0 = get_absolute_path(
        f"{cfg.models.polar.all}/partial_iso_molecules_models.sav"
    )
    path_partial_iso_mol_x1 = get_absolute_path(
        f"{cfg.models.polar.ei_alpha_dipole}/partial_iso_molecules_models.sav"
    )
    path_partial_iso_mol_x2 = get_absolute_path(
        f"{cfg.models.polar.alpha_dipole}/partial_iso_molecules_models.sav"
    )
    path_partial_iso_mol_x3 = get_absolute_path(
        f"{cfg.models.polar.alpha_dipole_pi}/partial_iso_molecules_models.sav"
    )

    list_of_paths = [
        path_partial_iso_mol_x0,
        path_partial_iso_mol_x1,
        path_partial_iso_mol_x2,
        path_partial_iso_mol_x3,
    ]

    list_of_models = []

    for i in list_of_paths:
        list_of_models.append(pickle.load(open(i, "rb")))

    for i, j in enumerate(list_of_feat_names):
        create_graph_shap(
            list_of_models[i],
            list_of_x_all[i],
            y_partial.ravel(),
            j,
            list_of_img_paths[i],
            "partial_iso_molecules_FI.png",
        )

    # For partial aniso molecules.

    x0_partial_aniso = df_polar_partial[
        cfg.opt.features.partial.polar.feat1_aniso
    ].values
    x1_partial_aniso = df_polar_partial[
        cfg.opt.features.partial.polar.feat2_aniso
    ].values
    x2_partial_aniso = df_polar_partial[
        cfg.opt.features.partial.polar.feat3_aniso
    ].values
    x3_partial_aniso = df_polar_partial[
        cfg.opt.features.partial.polar.feat4_aniso
    ].values

    list_of_x_all = [
        x0_partial_aniso,
        x1_partial_aniso,
        x2_partial_aniso,
        x3_partial_aniso,
    ]

    list_of_feat_names = [
        ["Ei", "axx", "ayy", "azz", "Dipole", "Pi Bond"],
        ["Ei", "axx", "ayy", "azz", "Dipole"],
        ["axx", "ayy", "azz", "Dipole"],
        ["axx", "ayy", "azz", "Dipole", "Pi Bond"],
    ]
    path_partial_aniso_mol_x0 = get_absolute_path(
        f"{cfg.models.polar.all}/partial_aniso_molecules_models.sav"
    )
    path_partial_aniso_mol_x1 = get_absolute_path(
        f"{cfg.models.polar.ei_alpha_dipole}/partial_aniso_molecules_models.sav"
    )
    path_partial_aniso_mol_x2 = get_absolute_path(
        f"{cfg.models.polar.alpha_dipole}/partial_aniso_molecules_models.sav"
    )
    path_partial_aniso_mol_x3 = get_absolute_path(
        f"{cfg.models.polar.alpha_dipole_pi}/partial_aniso_molecules_models.sav"
    )

    list_of_paths = [
        path_partial_aniso_mol_x0,
        path_partial_aniso_mol_x1,
        path_partial_aniso_mol_x2,
        path_partial_aniso_mol_x3,
    ]

    list_of_models = []

    for i in list_of_paths:
        list_of_models.append(pickle.load(open(i, "rb")))

    for i, j in enumerate(list_of_feat_names):
        create_graph_shap(
            list_of_models[i],
            list_of_x_all[i],
            y_partial.ravel(),
            j,
            list_of_img_paths[i],
            "partial_aniso_molecules_FI.png",
        )


if __name__ == "__main__":
    main()
