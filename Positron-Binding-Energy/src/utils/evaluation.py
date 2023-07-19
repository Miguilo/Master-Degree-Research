import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_validate


def display_scores(scores):
    """
    scores : Teste Scores of the cross_val procedure
    """

    print("Test Scores:", scores)
    print(f"Mean Test Scores: {np.round(scores.mean(), 2)}")
    print(f"Std of Scores: {np.round((scores.std()),2)}")
    score_min = np.round(scores.min(), 2)
    score_max = np.round(scores.max(), 2)
    print(f"Min and Max of Scores: {score_min}, {score_max}")


def get_cross_validation_scores(estimator, X, y, cv):

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
    """Algoritmo que devolve a análise da métrica a ser analisada em problema
    de regressão pra um data set já finalizado.

    scoring : The score that you want to show the metrics."""
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    train_score, test_score = get_cross_validation_scores(estimator, x, y, kf)

    display_scores(test_score[scoring])
    print(f"\nTrain Score: {np.round(np.mean(train_score[scoring]), 2)}\n")


def unity_norm(array):
    new_array = array / np.abs(array).sum()
    return new_array


def FI_shap_values(estimator, x, y):
    model = clone(estimator)
    model.fit(x, y)

    explainer = shap.KernelExplainer(model.predict, data=x)

    shap_values = explainer.shap_values(x)
    feature_importance = np.abs(shap_values).mean(axis=0)
    feature_importance = np.array(feature_importance)
    # print(feature_importance.shape)

    try:
        for i in range(feature_importance.shape[1]):
            feature_importance[:, i] = np.mean(feature_importance[:, i])
        return list(unity_norm(feature_importance[0]))
    except IndexError:  # Isso foi necessário pois de alguma forma, em alguns feature importance é gerado só 1 linha na matriz.
        return list(unity_norm(feature_importance))


def create_fast_graph(
    df,
    img_name,
    isotropy=True,
    img_path=None,
    y="Relative Error",
    palette="hls",
    title="Title",
    show_values=True,
    show_mean=True,
    figsize=(16, 9),
):

    columns = list(df.columns)
    for i in range(len(columns)):
        if "Alpha" in columns[i]:
            if isotropy:
                columns[i] = columns[i].replace("Alpha", "Isotropy")
            else:
                columns[i] = columns[i].replace("Alpha", "Anisotropy")

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

    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=dict_sns, x="Model", y=y, hue="Feat_Comparison", palette=palette
    )

    ax.set_xlabel("Models")
    ax.set_ylabel(f"{y}")
    ax.set_title("")

    if show_values:
        for p in ax.patches:
            ax.annotate(
                "%.2f" % p.get_height(),
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                fontsize=11,
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
    )
    if img_path != None:
        if not os.path.exists(img_path):
            os.mkdir(img_path)

        plt.savefig(f"{img_path}/{img_name}")


def plot_all_fast_graphs(list_of_df, img_path, img_names):

    for i, j in enumerate(list_of_df):
        if "aniso" in img_names[i]:
            create_fast_graph(
                j, isotropy=False, img_path=img_path, img_name=img_names[i]
            )
        else:
            create_fast_graph(j, img_path=img_path, img_name=img_names[i])


def create_mean_results(dict, features, error_column):
    t_df = pd.DataFrame(dict)
    mean_errors = []
    feature = []
    model = []
    for i in features:
        error = t_df.loc[t_df["Feature"] == i][error_column].mean()
        mean_errors.append(error)
        feature.append(i)
        model.append("Mean Error")
    new_dict = {
        "Feature Importance": mean_errors,
        "Models": model,
        "Feature": feature,
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
    models_names=["svr", "xgb", "nn", "poli", "stk"],
    figsize=(16, 9),
    title="SHAP Feature Importance",
    show_mean_error=True,
):
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
        "Feature": feat_column,
    }
    if show_mean_error:
        final_df = create_mean_results(
            plot_dict, feature_names, "Feature Importance"
        )
    else:
        final_df = plot_dict

    plt.figure(figsize=figsize)
    ax = sns.barplot(
        final_df,
        x="Models",
        y="Feature Importance",
        hue="Feature",
        palette="hls",
    )
    ax.set_title(title)
    ax.set_xlabel("Models")
    ax.set_ylabel("Feature Importance")

    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    plt.savefig(f"{path_to_save}/{img_name}")
