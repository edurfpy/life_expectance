# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

# valores colormap
TABLEAU_CMP = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
               'tab:olive', 'tab:cyan')


def resultados_GSearchCV(results_gridcv, n_first=5, metric_descrip=None):
    # filtrado y orden resultados
    resultados = pd.DataFrame(results_gridcv)
    resultados.mean_test_score = np.abs(resultados.mean_test_score)
    resultados.mean_train_score = np.abs(resultados.mean_train_score)
    resultados = resultados.filter(regex='(param.*|mean_t|std_t)') \
        .sort_values('mean_test_score', ascending=True) \
        .head(n_first)

    # orden inverso datos grafica, para correspondencia con tabla
    parametros = resultados['params'].to_list()
    parametros.reverse()
    mean_test = np.flip(resultados['mean_test_score'])
    std_test = np.flip(resultados['std_test_score'])

    # grafica resultados

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.barh([str(d) for d in parametros], mean_test, xerr=std_test, align='center', alpha=0)
    ax.plot(mean_test, [str(d) for d in parametros], marker="D", linestyle="", alpha=0.8, color="r")

    ax.set_title('Comparación de Hiperparámetros')

    xlabel = f'Test score ({metric_descrip})' if metric_descrip else 'Test score'
    ax.set_xlabel(xlabel)

    return resultados.drop(columns='params')


def plot_feature_importance(column_names, model_importance, model_based_type='tree', fig_size=(10, 12)):
    if model_based_type == 'tree':
        model_importance = model_importance / model_importance.max()

    color_sign = ['pos' if valor >= 0 else 'neg' for valor in model_importance]
    df_feat_imp = pd.DataFrame({'features': column_names, 'importance': model_importance,
                                'color': color_sign})
    df_feat_imp = df_feat_imp.sort_values('importance', ascending=False)

    fig, axes = plt.subplots(figsize=fig_size)
    sns.barplot(x='importance', y='features', hue='color', data=df_feat_imp, ax=axes)

    sns.move_legend(axes, "lower right")
    axes.set_title('VARIABLE IMPORTANCE')

    if model_based_type == 'tree':
        axes.set_xlabel('Relative Importance')

    elif model_based_type == 'linear':
        axes.set_xlabel('COEFS VALUES')
