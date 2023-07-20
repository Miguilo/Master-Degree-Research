Module src.utils.evaluation
===========================

Functions
---------

    
`FI_shap_values(estimator, x, y)`
:   

    
`create_fast_graph(df, img_name, isotropy=True, img_path=None, y='Relative Error', palette='hls', title='Title', show_values=True, show_mean=True, figsize=(16, 9))`
:   

    
`create_graph_shap(estimators, x, y, feature_names, path_to_save, img_name, models_names=['poly', 'svr', 'xgb', 'nn', 'stk'], figsize=(16, 9), title='SHAP Feature Importance', show_mean_error=True)`
:   

    
`create_mean_results(dict, features, error_column)`
:   

    
`display_scores(scores)`
:   scores : Teste Scores of the cross_val procedure

    
`get_cross_validation_scores(estimator, X, y, cv)`
:   

    
`plot_all_fast_graphs(list_of_df, img_path, img_names)`
:   

    
`show_metrics(estimator, x, y, cv, scoring='neg_mean_absolute_percentage_error')`
:   Algoritmo que devolve a análise da métrica a ser analisada em problema
    de regressão pra um data set já finalizado.
    
    scoring : The score that you want to show the metrics.

    
`unity_norm(array)`
: