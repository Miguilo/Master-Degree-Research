import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_validate

legend_fontsize = 20
tick_fontsize = 17
label_fontsize = 20
annotate_fontsize = 13

title_size = 22


def display_scores(scores):
    """
    Displays the scores of the cross-validation procedure.

    Args:
        scores: The test scores of the cross-validation procedure.

    Returns:
        None.

    """

    print("Test Scores:", scores)
    print(f"Mean Test Scores: {np.round(scores.mean(), 2)}")
    print(f"Std of Scores: {np.round((scores.std()),2)}")
    score_min = np.round(scores.min(), 2)
    score_max = np.round(scores.max(), 2)
    print(f"Min and Max of Scores: {score_min}, {score_max}")


def get_cross_validation_scores(estimator, X, y, cv):
    """

    Get the cross-validation scores for a given estimator.

    Args:
        estimator: The estimator to be evaluated.
        X: The training data.
        y: The target values.
        cv: The cross-validation folds.

    Returns:
        A tuple of train and test scores.

    """

    scoring = [
        "neg_root_mean_squared_error",
        "neg_mean_absolute_percentage_error",
    ]
    cv_results = cross_validate(
        estimator, X, y, cv=cv, scoring=scoring, return_train_score=True
    )

    train_scores = {
        "neg_root_mean_squared_error": -cv_results[
            "train_neg_root_mean_squared_error"
        ],
        "neg_mean_absolute_percentage_error": -cv_results[
            "train_neg_mean_absolute_percentage_error"
        ],
    }

    test_scores = {
        "neg_root_mean_squared_error": -cv_results[
            "test_neg_root_mean_squared_error"
        ],
        "neg_mean_absolute_percentage_error": -cv_results[
            "test_neg_mean_absolute_percentage_error"
        ],
    }

    return train_scores, test_scores


def show_metrics(
    estimator, x, y, cv, scoring="neg_mean_absolute_percentage_error"
):
    """
    Displays the metrics of a regression model.

    Args:
        estimator: The estimator to be evaluated.
        x: The training data.
        y: The target values.
        cv: The cross-validation folds.
        scoring: The scoring metric.

    Returns:
        None.

    """
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    train_score, test_score = get_cross_validation_scores(estimator, x, y, kf)

    display_scores(test_score[scoring])
    print(f"\nTrain Score: {np.round(np.mean(train_score[scoring]), 2)}\n")


def unity_norm(array):
    new_array = array / np.abs(array).sum()
    return new_array


def FI_shap_values(estimator, x, y):
    """
    Computes the feature importance using SHAP values.

    Args:
        estimator: The estimator to be evaluated.
        x: The training data.
        y: The target values.

    Returns:
        A list of feature importance scores.

    """
    model = clone(estimator)
    model.fit(x, y)

    explainer = shap.KernelExplainer(model.predict, data=x)

    shap_values = explainer.shap_values(x)
    print(shap_values)
    feature_importance = np.abs(shap_values).mean(axis=0)
    feature_importance = np.array(feature_importance)
    print(feature_importance)

    try:
        for i in range(feature_importance.shape[1]):
            feature_importance[:, i] = np.mean(feature_importance[:, i])
        return list(unity_norm(feature_importance[0]))
    except IndexError:  # Isso foi necessário pois de alguma forma, em alguns feature importance é gerado só 1 linha na matriz.
        return list(unity_norm(feature_importance))


def get_sub(x):
    normal = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    )
    sub_s = (
        "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    )
    res = x.maketrans("".join(normal), "".join(sub_s))
    return x.translate(res)


def create_fast_graph(
    df,
    img_name,
    isotropy=True,
    img_path=None,
    y="Erro Percentual Absoluto Médio",
    palette="hls",
    show_values=True,
    show_mean=True,
    figsize=(16, 9),
    partial=False,
    tick_fontsize=tick_fontsize,
    label_fontsize=label_fontsize,
    annotate_fontsize=annotate_fontsize,
    legend_fontsize=legend_fontsize,
):
    """
    Create a barplot for the error of different models.

    Args:
        df: The dataframe with the errors.
        img_name: The name of the image to be saved.
        isotropy: Whether the error is for isotropy or anisotropy.
        img_path: The path where the image will be saved.
        y: The name of the column with the error values.
        palette: The palette to be used in the barplot.
        title: The title of the barplot.
        show_values: Whether to show the values on the bars.
        show_mean: Whether to show the mean error in the barplot.
        figsize: The size of the barplot.

    Returns:
        None.

    """

    columns = list(df.columns)
    for i in range(len(columns)):
        if "Ei" in columns[i]:
            columns[i] = columns[i].replace("Ei", r"$I_{P}$")

        if "Alpha" in columns[i]:
            if isotropy and not partial:
                columns[i] = columns[i].replace("Alpha", r"$\alpha$")
            elif isotropy and partial:
                columns[i] = columns[i].replace("Alpha", r"$\bar\alpha$")
            else:
                columns[i] = columns[i].replace(
                    "Alpha", r"$\alpha_{(xx, yy, zz)}$"
                )

        if "Pi" in columns[i]:
            columns[i] = columns[i].replace("Pi", r"$\pi$")

        if "Dipole" in columns[i]:
            columns[i] = columns[i].replace("Dipole", r"$\mu$")

    hue = columns
    models_name = list(df.index)

    list_of_errors = []
    for i in df.columns:
        list_of_errors.append(df[i].values)

    model_list = []
    hue_list = []
    errors = []
    aux_model = models_name.copy()

    if show_mean:
        aux_model.append("Mean Error")

    for i in list_of_errors:
        errors.extend(i)
        if show_mean:
            errors.append(np.mean(i))

    while len(model_list) < len(errors):
        model_list.extend(aux_model)

    j = 0
    while len(hue_list) < len(errors):
        hue_list.extend([hue[j]] * len(aux_model))
        j = (j + 1) % len(hue)

    dict_sns = {
        f"{y}": errors,
        "Model": model_list,
        "Feat_Comparison": hue_list,
    }

    dict_for_map = {
        "svr": "SVR",
        "nn": "RNA",
        "xgb": "XGBoost",
        "poly": "Regressão Polinomial",
        "stacked": "Modelo Ensemble",
        "Mean Error": "Erro Médio",
    }

    for i, j in enumerate(dict_sns["Model"]):
        if j in dict_for_map.keys():
            dict_sns["Model"][i] = dict_for_map[j]

    plt.figure(figsize=figsize, dpi=300)

    ax = sns.barplot(
        data=dict_sns, x="Model", y=y, hue="Feat_Comparison", palette=palette
    )

    ax.set_xlabel("Modelos", fontsize=label_fontsize)
    ax.set_ylabel(f"{y}", fontsize=label_fontsize)
    ax.set_title("")

    ax.tick_params(axis="both", labelsize=tick_fontsize)

    if show_values:
        for p in ax.patches:
            ax.annotate(
                "%.2f" % p.get_height(),
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                fontsize=annotate_fontsize,
                color="gray",
                xytext=(0, 10),
                textcoords="offset points",
            )

    sns.move_legend(
        ax,
        "lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=len(hue),
        title=None,
        frameon=False,
        fontsize=legend_fontsize,
    )
    if img_path != None:
        if not os.path.exists(img_path):
            os.mkdir(img_path)

        plt.savefig(f"{img_path}/{img_name}")


def plot_all_fast_graphs(list_of_df, img_path, img_names, partial=False):

    for i, j in enumerate(list_of_df):
        if "partial" in img_names[i]:
            partial = True

        if "aniso" in img_names[i]:
            create_fast_graph(
                j,
                isotropy=False,
                img_path=img_path,
                img_name=img_names[i],
                partial=partial,
            )
        else:
            create_fast_graph(
                j, img_path=img_path, img_name=img_names[i], partial=partial
            )


def create_mean_results(dict, features, error_column):
    t_df = pd.DataFrame(dict)
    mean_errors = []
    feature = []
    model = []
    for i in features:
        error = t_df.loc[t_df["Propriedade Molecular"] == i][
            error_column
        ].mean()
        mean_errors.append(error)
        feature.append(i)
        model.append("Média")
    new_dict = {
        "Feature Importance": mean_errors,
        "Models": model,
        "Propriedade Molecular": feature,
    }
    new_df = pd.DataFrame(new_dict)
    final_df = pd.concat([t_df, new_df], ignore_index=True)

    return final_df


def create_graph_shap(
    estimators,
    x,
    y,
    feature_names,
    path_to_save,
    img_name,
    models_names=[
        "Regressão Polinomial",
        "SVR",
        "XGBoost",
        "RNA",
        "Modelo Ensemble",
    ],
    figsize=(16, 9),
    title="Importância de Propriedades via SHAP",
    show_mean_error=True,
    tick_fontsize=tick_fontsize,
    label_fontsize=label_fontsize,
    legend_fontsize=legend_fontsize,
    title_size=title_size,
):
    """
    Create a barplot for the SHAP feature importance of different models.

    Args:
        estimators: The list of estimators.
        x: The training data.
        y: The target values.
        feature_names: The names of the features.
        path_to_save: The path where the image will be saved.
        img_name: The name of the image to be saved.
        models_names: The names of the models.
        figsize: The size of the barplot.
        title: The title of the barplot.
        show_mean_error: Whether to show the mean error in the barplot.

    Returns:
        None.
    """
    scores = []
    feat_column = []
    models = []
    for i, j in zip(estimators, models_names):
        # print('entered the loop')
        feat_score = FI_shap_values(i, x, y)
        scores.extend(feat_score)
        feat_column.extend(feature_names)
        list_models_names = [j]
        models.extend(list_models_names * len(feature_names))

    plot_dict = {
        "Feature Importance": scores,
        "Models": models,
        "Propriedade Molecular": feat_column,
    }

    if show_mean_error:
        final_df = create_mean_results(
            plot_dict, feature_names, "Feature Importance"
        )
    else:
        final_df = plot_dict

    dict_for_map = {
        "Ei": r"$I_{P}$",
        "Pi Bond": r"Ligações $\pi$",
        "Dipole": r"$\mu$",
        "axx": r"$\alpha_{xx}$",
        "ayy": r"$\alpha_{yy}$",
        "azz": r"$\alpha_{zz}$",
    }

    if "partial" in img_name:
        dict_for_map["Alpha"] = r"$\bar\alpha$"
    else:
        dict_for_map["Alpha"] = r"$\alpha$"

    for i, j in enumerate(final_df["Propriedade Molecular"]):
        if j in dict_for_map.keys():
            final_df["Propriedade Molecular"][i] = dict_for_map[j]

    plot_df = pd.DataFrame(plot_dict)

    plt.figure(figsize=figsize, dpi=300)
    ax = sns.barplot(
        final_df,
        x="Models",
        y="Feature Importance",
        hue="Propriedade Molecular",
        palette="hls",
    )
    ax.set_title(title, fontsize=title_size)
    ax.set_xlabel("Modelos", fontsize=label_fontsize)
    ax.set_ylabel(
        "Importância Percentual Relativa de Propriedades",
        fontsize=label_fontsize,
    )

    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize)

    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    plt.savefig(f"{path_to_save}/{img_name}")
    img_to_csv = img_name.replace("png", "csv")
    plot_df.to_csv(f"{path_to_save}/{img_to_csv}", index_label=False)
