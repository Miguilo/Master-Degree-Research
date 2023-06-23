import numpy as np
from sklearn.model_selection import cross_validate


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
