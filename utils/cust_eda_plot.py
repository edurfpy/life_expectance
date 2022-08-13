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


def plot_unidim_num(df,columns,file_save=None):

    # creamos la grafica
    fig, axes = plt.subplots(len(columns), 2, figsize=(20, 7 * len(columns)), gridspec_kw={'hspace': 0.4})
    ax = axes.ravel()

    # graficas distribucion y boxplot de cada atributo
    for idx, atributo in enumerate(columns):

        # distribucion (histograma)
        sns.distplot(df[atributo], bins=30, ax=ax[2 * idx], color=TABLEAU_CMP[idx % len(TABLEAU_CMP)],
                     hist_kws={'alpha': 0.15})
        # hist_kws={'alpha': 0.15}

        # titulo, etiquetas histograma
        ax[2 * idx].set_title(atributo)
        ax[2 * idx].set_xlabel(f"Valores atributo [missing: {df[atributo].isnull().sum()} ({np.round(df[atributo].isnull().mean()*100,2)} %)]")
        ax[2 * idx].set_ylabel("Frequencia")

        # boxplot
        sns.boxplot(x=atributo, data=df, ax=ax[2 * idx + 1], color=TABLEAU_CMP[idx % len(TABLEAU_CMP)])

        # titulo, etiquetas boxplot
        ax[2 * idx + 1].set_title(atributo)
        ax[2 * idx + 1].set_xlabel("")


    # guardamos grafica:
    if file_save:
        fig.savefig(file_save)



def plot_unidim_cat(df,columnms,file_save=None):

    if len(columnms) == 1:

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 7), gridspec_kw={'hspace': 0.4})
        sns.countplot(data=df, x=columnms[0], ax=axes)
        axes.set_title(columnms[0])
        axes.set_xlabel(f"Valores atributo [missing: {df[columnms[0]].isnull().sum()} ({np.round(df[columnms[0]].isnull().mean() * 100, 2)} %)]")
        axes.set_ylabel("Frequencia")

    else:

        # nº filas del 'grid' para 2 columnas
        nrows = int(len(columnms) / 2) if len(columnms) % 2 == 0 else int(len(columnms) // 2 + 1)

        # creamos la grafica
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(20, 7 * nrows), gridspec_kw={'hspace': 0.4})

        ax = axes.ravel()
        if len(columnms) % 2 != 0: ax[-1].axis('off')

        # graficas de barras de cada atributo
        for idx, atributo in enumerate(columnms):
            # distribucion (histograma)
            sns.countplot(data=df, x=atributo, ax=ax[idx])

            # titulo, etiquetas histograma
            ax[idx].set_title(atributo)
            ax[idx].set_xlabel(f"Valores atributo [missing: {df[atributo].isnull().sum()} ({np.round(df[atributo].isnull().mean()*100,2)} %)]")
            ax[idx].set_ylabel("Frequencia")


    # guardamos grafica:
    if file_save:
        fig.savefig(file_save)




def plot_matrx_corr(df,columns,file_save=None):

    # crear matriz correlacion
    corrmat = np.round(df[columns].corr(), decimals=2)

    # crear heatmap
    fig_size = np.ceil(0.75*len(columns))
    plt.figure(figsize=(fig_size, fig_size))
    ax = sns.heatmap(corrmat, annot=True, cbar=True, cmap='RdBu_r', fmt='.3g', square=True)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    fig = ax.get_figure()

    # guardamos grafica:
    if file_save:
        fig.savefig(file_save)




def plot_scttreg_target(df,features,target,file_save=None):

    corrmat = df[features + [target]].corr()

    nrows = int(len(features) / 2) if len(features) % 2 == 0 else int(len(features) // 2 + 1)

    # creamos la grafica
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(20, 7 * nrows), gridspec_kw={'hspace': 0.4})

    ax = axes.ravel()
    if len(features) % 2 != 0: ax[-1].axis('off')

    # scatterplot de cada atributo con precio
    for idx, atributo in enumerate(features):
        # scatterplot (regresion)
        sns.regplot(data=df, x=atributo, y=target, ax=ax[idx], color='blue', marker='.',
                    scatter_kws={"alpha": 0.4},
                    line_kws={"color": "r", "alpha": 0.7})

        # titulo, etiquetas histograma
        ax[idx].set_title(f'{atributo}: r = {np.round(corrmat.loc[atributo,target],3)}')
        ax[idx].set_xlabel(atributo)
        ax[idx].set_ylabel(target)

    # guardamos grafica:
    if file_save:
        fig.savefig(file_save)



def plot_categ_target(df, features, target, file_save=None):

    if len(features) == 1:

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 7), gridspec_kw={'hspace': 0.4})
        sns.violinplot(data=df, x=features[0], y=target, ax=axes)
        axes.set_title(f'{features[0]} vs {target}')
        axes.set_xlabel(features[0])
        axes.set_ylabel(target)

    else:

        # nº filas del 'grid' para 2 columnas
        nrows = int(len(features) / 2) if len(features) % 2 == 0 else int(len(features) // 2 + 1)

        # creamos la grafica
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(20, 7 * nrows), gridspec_kw={'hspace': 0.4})

        ax = axes.ravel()
        if len(features) % 2 != 0: ax[-1].axis('off')

        # graficas de barras de cada atributo
        for idx, atributo in enumerate(features):
            # distribucion (histograma)
            sns.violinplot(data=df, x=atributo, y=target, ax=ax[idx])

            # titulo, etiquetas histograma
            axes.set_title(f'{atributo} vs {target}')
            axes.set_xlabel(atributo)
            axes.set_ylabel(target)


    # guardamos grafica:
    if file_save:
        fig.savefig(file_save)




























