import copy
import pickle

import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from skopt import gp_minimize
from skopt.space.space import Categorical, Integer, Real
from skopt.utils import use_named_args
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from IPython.display import clear_output
from sklearn.base import clone

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.base import clone

def modify_scaling(list_of_models, list_of_models_names, name_of_features):
    modified_models = []  # Lista para armazenar os modelos modificados
    
    for model, name in zip(list_of_models, list_of_models_names):
        if 'xgb' in name:
            modified_models.append(model)
        else:
            ct = ColumnTransformer([('scale', RobustScaler(), slice(0, None))], remainder='passthrough')
            if 'pi' in name_of_features.lower() or 'all' in name_of_features.lower():
                ct = ColumnTransformer([('scale', RobustScaler(), slice(0, -1))], remainder='passthrough')
            elif 'poly' in name.lower() or 'poli' in name.lower():
                ct = ColumnTransformer([('scale', RobustScaler(), slice(0, None))], remainder='passthrough')
            
            modified_model = clone(model)  
            modified_model.set_params(regressor__scale=ct) 
            modified_models.append(modified_model)  
        
    return modified_models



def convert_to_space(caller, parameter):
    """
    caller: Should have an form of cfg.opt.estimator
    parameter: Key of the hyperparameter in the way it's write
    in .yaml file.
    """
    param_info = caller[parameter]
    name = param_info["name"]
    type = param_info["type"]

    if "low" in param_info:
        low = param_info["low"]
        high = param_info["high"]

        if type == "real":
            if "prior" in param_info:
                return Real(low, high, prior=param_info["prior"], name=name)
            else:
                return Real(low, high, name=name)

        else:
            return Integer(low, high, name=name)

    elif type == "scalers":
        return Categorical(
            [eval(i) for i in param_info["categories"]], name=name
        )

    else:
        return Categorical(param_info["categories"], name=name)


def change_nn_params(params):
    n_neurons = params["n_neurons_per_layer"]
    n_layers = params["n_hidden_layer"]

    params["regressor__reg__hidden_layer_sizes"] = (n_neurons,) * n_layers

    # the parameters are deleted to avoid an error from the MLPRegressor
    params.pop("n_neurons_per_layer")
    params.pop("n_hidden_layer")


def att_model(estimator, space, values, neural=False):
    """
    It is important to remember that in the case of neural networks,
    the name of the
    parameters associated with hidden_layers must be
    "n_neurons_per_layer"
    and "n_hidden_layer"
    """
    params = {}

    for i in range(len(space)):
        params[f"{space[i].name}"] = values[i]
    if neural:
        change_nn_params(params)

    estimator.set_params(**params)

    return estimator


def gp_optimize(
    estim,
    x,
    y,
    space,
    cv,
    n_calls=150,
    n_random_starts=100,
    neural=False,
    scoring="neg_mean_absolute_percentage_error",
    n_jobs_cv=6,
    n_jobs_opt=6,
    verbose=0,
):
    @use_named_args(space)
    def objective(**params):
        if neural:
            change_nn_params(params)

        estim.set_params(**params)

        mean_score = -np.mean(
            cross_val_score(
                estim,
                x,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs_cv,
                error_score=-100,
            )
        )

        if mean_score > 1000:
            mean_score = 100

        return mean_score

    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        random_state=0,
        n_random_starts=n_random_starts,
        verbose=verbose,
        n_jobs=n_jobs_opt,
    )
    # print("Os hiper parâmetros que minimizam são: ", result.x)
    return result


def save_models(path: str, model):
    pickle.dump(model, open(path, "wb"))


def opt_all(
    estimator_list,
    names_estimator_list,
    spaces_list,
    x,
    y,
    cv,
    path,
    verbose=0,
    print_models=False
):
    list_of_models = []
    for j, k in enumerate(estimator_list):
        neural = False
        if verbose != 0:
            print(f"--- We're in the model {names_estimator_list[j]} ---")

        if "nn" in names_estimator_list[j]:
            neural = True

        opt = gp_optimize(k, x, y, spaces_list[j], cv=cv, neural=neural)
        best_model = att_model(k, spaces_list[j], opt.x, neural=neural)
        copy_of_best_model = copy.deepcopy(
            best_model
        )  # To ensure that we're not changing the original estimator
        trained_model = copy_of_best_model.fit(x, y)
        list_of_models.append(trained_model)

    stk_model = VotingRegressor(
        [(a, b) for a, b in zip(names_estimator_list, list_of_models)]
    )
    copy_of_stk = copy.deepcopy(stk_model)
    trained_stk = copy_of_stk.fit(x, y)  # The same as in line 152
    list_of_models.append(trained_stk)

    if print_models:
        for i in list_of_models:
            print(i)

    save_models(path, list_of_models)

def nested_cv(estimator, space, x, y, out_cv, inner_cv, scoring = "neg_mean_absolute_percentage_error",
               neural = False, n_calls=15, n_random_starts = 10, verbose = 0, shuffle=True, 
               random_state=0, print_mode = False):
    
    '''
    A função retornará os scores de cada fold pra k_fold externo
    train_scores: Score de treino pra cada fold de treino.
    test_pred: Predição de teste pra cada elemento quando esteve no conjunto de teste.
    test_error: Erro de teste pra cada elemento quando esteve no conjunto de teste.
    cv_score: Erro de teste de cada k_fold externo.
    stacked: Argumento utilizado quando o modelo a ter o nested-cv retirado é um modelo stacking.

    Importante lembrar que a média do test error não será equivalente ao cv score.
    Isto pois o agrupamento de dados de teste é diferente do que a simples soma e divisão por N
    da média do test error.
    '''

    kf_out = KFold(n_splits = out_cv, shuffle=shuffle, random_state=random_state) #Folders externo.
    kf_in = KFold(n_splits = inner_cv, shuffle=shuffle, random_state=random_state) #Folders internos.

    test_pred = np.zeros_like(y)
    test_error = np.zeros_like(y)

    cv_scores = []
    train_scores = []
    count = 1

    #Dividindo entre treino e teste na pasta de fora.
    for train_index, test_index in kf_out.split(x):
        print(f"Out folder {count} of {out_cv}.")
        count += 1
        x_train, x_test = x[list(train_index)], x[list(test_index)]
        y_train, y_test = y[train_index], y[test_index]
        
        #Otimizando o modelo pro x_train e y_train com validação cruzada interna.
        opt_model = gp_optimize(estimator, x_train, y_train, space = space, cv = kf_in, n_calls=n_calls, 
                                n_random_starts = n_random_starts, 
                                neural = neural, scoring = scoring, verbose = verbose)
        
        #Pegando o melhor modelo com os valores de otimização
        att_model(estimator, space, opt_model.x, neural = neural)
        #Fitando o melhor modelo
        fitted_model = estimator.fit(x_train, y_train)
        #Prevendo agora com o melhor modelo, a performance no dado de teste.
        y_train_pred = fitted_model.predict(x_train)
        y_pred = fitted_model.predict(x_test)


        #Mudando a métrica de acordo com scoring
        if scoring == "neg_mean_absolute_percentage_error":
            cv_scores.append(mean_absolute_percentage_error(y_test, y_pred))
            train_scores.append(mean_absolute_percentage_error(y_train,y_train_pred))
        elif scoring == "neg_root_mean_squared_error":
            cv_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            train_scores.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))

        #Gerando o erro de teste índice a índice
        for i,z,w in zip(test_index, x_test, y_test):
            test_pred[i] = fitted_model.predict(z.reshape(1,-1))
            if scoring == "neg_mean_absolute_percentage_error":
                test_error[i] = mean_absolute_percentage_error(w, test_pred[i])
            elif scoring=="neg_root_mean_squared_error":
                test_error[i] = np.sqrt(mean_squared_error(w, test_pred[i]))
  

    if print_mode:
        clear_output()
        print("Cv Score:", np.mean(cv_scores))
        print("Desvio Padrão:", round(np.std(cv_scores),2))
        print("Score de Treino:", np.mean(train_scores))

    return np.mean(cv_scores), round(np.std(cv_scores),2), np.mean(train_scores), test_error


def stacked_nested_cv(estimators, estimators_names, spaces, x, y, out_cv, inner_cv, scoring = "neg_mean_absolute_percentage_error",
               neural = False, n_calls=15, n_random_starts = 10, verbose = 0, shuffle=True, 
               random_state=0):
    '''
    A função retornará os scores de cada fold pra k_fold externo em um modelo stacked via voting
    estimators: Lista contendo estimadores a serem otimizados para futura agregação no stacking.
    Importante lembrar que a média do test error não será equivalente ao cv score.
    Isto pois o agrupamento de dados de teste é diferente do que a simples soma e divisão por N
    da média do test error.
    - Params
    estimators_names: Lista contendo string dos nomes dos estimadores a serem otimizados.
    spaces: Espaços de busca de hiperparâmetro individual pra cada estimador.
    '''
    kf_out = KFold(n_splits = out_cv, shuffle=shuffle, random_state=random_state) #Folders externos.
    kf_in = KFold(n_splits = inner_cv, shuffle=shuffle, random_state=random_state) #Folders internos.

    dict_test_scores = {} # Dicionário que conterá os scores de teste de cada modelo
    dict_train_scores = {} # Dicionário que conterá os scores de treino de cada modelo
    for i in estimators_names:
        dict_test_scores[i] = [] # Adicionando as chaves por nome do modelo
        dict_train_scores[i] = [] # Adicionando as chaves por nome do modelo
        dict_test_scores['stacked'] = []
        dict_train_scores['stacked'] = []

    count = 1
    
    #Dividindo entre treino e teste na pasta de fora.
    for train_index, test_index in kf_out.split(x):
        list_of_models = [] # Lista que conterá os modelos atualizados para pilhar no VotingRegressor
        print(f"Out folder {count} of {out_cv}.")
        count += 1
        x_train, x_test = x[list(train_index)], x[list(test_index)]
        y_train, y_test = y[train_index], y[test_index]
        
        #Otimizando o modelo pro x_train e y_train com validação cruzada interna.
        for i,j in enumerate(estimators):
            print(f"Optimizing estimator {estimators_names[i]}")
            neural=False # Garantindo que não haja um modelo após uma rede neural com parâmetro "neural = True"
            if 'nn' in estimators_names[i].lower():
                neural=True
            opt_model = gp_optimize(j, x_train, y_train, space = spaces[i], cv = kf_in, 
                                n_calls=n_calls, n_random_starts = n_random_starts, 
                                neural = neural, scoring = scoring, verbose = verbose)
        #Pegando o melhor modelo com os valores de otimização
            att_model(j, spaces[i], opt_model.x, neural = neural)
            list_of_models.append((estimators_names[i], j))

        stk_model = VotingRegressor(list_of_models, n_jobs=6)
        
        for i,j in list_of_models:
            # Fitando os modelos
            copied_model = clone(j)
            fitted_model = copied_model.fit(x_train, y_train)
            # Prevendo com os melhores modelos, a performance nos dados de treino e teste
            y_train_pred = fitted_model.predict(x_train)
            y_pred = fitted_model.predict(x_test)

            # Adicionando os scores de teste/treino nos dicionários por nome de modelo
            if scoring == "neg_mean_absolute_percentage_error":
                dict_test_scores[i].append(mean_absolute_percentage_error(y_test, y_pred))
                dict_train_scores[i].append(mean_absolute_percentage_error(y_train, y_train_pred))
            elif scoring == "neg_root_mean_squared_error":
                dict_test_scores[i].append(np.sqrt(mean_squared_error(y_test, y_pred)))
                dict_train_scores[i].append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
            
        # Já adicionando o score do modelo stacked
        stk_model.fit(x_train, y_train)
        y_pred_stk = stk_model.predict(x_test)
        y_train_pred_stk = stk_model.predict(x_train)
        if scoring == "neg_mean_absolute_percentage_error":
            dict_test_scores['stacked'].append(mean_absolute_percentage_error(y_test, y_pred_stk))
            dict_train_scores['stacked'].append(mean_absolute_percentage_error(y_train, y_train_pred_stk))
        elif scoring == "neg_root_mean_squared_error":
            dict_test_scores['stacked'].append(np.sqrt(mean_squared_error(y_test, y_pred_stk)))
            dict_train_scores['stacked'].append(np.sqrt(mean_squared_error(y_train, y_train_pred_stk)))

    return dict_test_scores, dict_train_scores
        
