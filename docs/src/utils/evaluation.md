Module src.utils.evaluation
===========================

Functions
---------

    
`FI_shap_values(estimator, x, y)`
: 

Computes the feature importance using SHAP values.
    
    Args:
        estimator: The estimator to be evaluated.
        x: The training data.
        y: The target values.
    
    Returns:
        A list of feature importance scores.

    
`create_fast_graph(df, img_name, isotropy=True, img_path=None, y='Relative Error', palette='hls', title='Title', show_values=True, show_mean=True, figsize=(16, 9))`
: 

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

    
`create_graph_shap(estimators, x, y, feature_names, path_to_save, img_name, models_names=['poly', 'svr', 'xgb', 'nn', 'stk'], figsize=(16, 9), title='SHAP Feature Importance', show_mean_error=True)`
:   

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
    
`display_scores(scores)`
:  

 Displays the scores of the cross-validation procedure.
    
    Args:
        scores: The test scores of the cross-validation procedure.
    
    Returns:
        None.

    
`get_cross_validation_scores(estimator, X, y, cv)`:

Get the cross-validation scores for a given estimator.
    
    Args:
        estimator: The estimator to be evaluated.
        X: The training data.
        y: The target values.
        cv: The cross-validation folds.
    
    Returns:
        A tuple of train and test scores.

    
`show_metrics(estimator, x, y, cv, scoring='neg_mean_absolute_percentage_error')`:
  
  Displays the metrics of a regression model.
    
    Args:
        estimator: The estimator to be evaluated.
        x: The training data.
        y: The target values.
        cv: The cross-validation folds.
        scoring: The scoring metric.
    
    Returns:
        None.