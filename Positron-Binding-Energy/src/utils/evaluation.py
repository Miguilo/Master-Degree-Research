import numpy as np
from sklearn.model_selection import cross_validate
import shap
from sklearn.base import clone
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

    train_score, test_score = get_cross_validation_scores(estimator, x, y, cv)

    display_scores(test_score[scoring])
    print(f"\nTrain Score: {np.round(np.mean(train_score[scoring]), 2)}\n")

def unity_norm(array):
    new_array = array/np.abs(array).sum()
    return new_array

def FI_shap_values(estimator, x, y):
    model=clone(estimator)
    model.fit(x,y)

    explainer = shap.KernelExplainer(model.predict, data=x)

    shap_values = explainer.shap_values(x)
    feature_importance = np.abs(shap_values).mean(axis=0)
    feature_importance = np.array(feature_importance)
    #print(feature_importance.shape)

    try:
        for i in range(feature_importance.shape[1]):
            feature_importance[:,i] = np.mean(feature_importance[:,i])
        return list(unity_norm(feature_importance[0]))
    except IndexError: #Isso foi necessário pois de alguma forma, em alguns feature importance é gerado só 1 linha na matriz.
        return list(unity_norm(feature_importance))
    
def create_mean_results(dict, features, error_column):
    t_df = pd.DataFrame(dict)
    mean_errors = []
    feature = []
    model = []
    for i in features:
        error = t_df.loc[t_df['Feature'] == i][error_column].mean()
        mean_errors.append(error)
        feature.append(i)
        model.append('Mean Error')
    new_dict = {
        'Feature Importance':mean_errors,
        'Models':model,
        'Feature':feature
    }
    new_df = pd.DataFrame(new_dict)
    final_df = pd.concat([t_df, new_df], ignore_index=True)

    return final_df

def create_graph_shap(estimators, x ,y, feature_names, models_names=['svr','xgb','nn','poli', 'stk'], 
                      figsize = (16,9), title = 'SHAP Feature Importance', show_mean_error = True):
    scores = []
    feat_column = []
    models = []
    for i,j in zip(estimators, models_names):
        #print('entered the loop')
        feat_score = FI_shap_values(i, x, y,j)
        scores.extend(feat_score)
        feat_column.extend(feature_names)
        list_models_names = [j]
        models.extend(list_models_names*len(feature_names))

    plot_dict = {
        'Feature Importance':scores,
        'Models':models,
        'Feature': feat_column
    }
    if show_mean_error:
        final_df = create_mean_results(plot_dict, feature_names, 'Feature Importance')
    else:
        final_df = plot_dict

    plt.figure(figsize=figsize)
    ax = sns.barplot(final_df, x='Models', y='Feature Importance', hue='Feature', palette='hls')
    ax.set_title(title)
    ax.set_xlabel("Models")
    ax.set_ylabel("Feature Importance")
    plt.show()