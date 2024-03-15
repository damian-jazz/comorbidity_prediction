
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plot_path = 'plots/'

def plot_demographics(df: pd.DataFrame, set_tag: str):
    #palette = 'bright' # 'Set2'
    fig = plt.figure(1,(14,4))

    g1 = plt.subplot(1,3,1)
    sns.histplot(x='Age', data=df, hue='Sex', bins=100, multiple="dodge")
    g1.set(ylabel=None)
    #plt.title("Age")

    #plt.subplot(2,2,2)
    #g2_labels = ['4.5 - 8.5','7.0 - 11.0','7.5 - 13.5','10.0 - 14.0','13.0 - 18.5']
    #g2 = sns.countplot(x='Cohort', data=df, palette=palette)
    #g2.legend(g2.containers[0], g2_labels, title='Age groups')
    #g2.bar_label(g2.containers[0], label_type='center')
    #g2.set(ylabel=None)
    #plt.title("Cohort")

    plt.subplot(1,3,2)
    g3 = sns.countplot(x='Sex', data=df)
    g3.bar_label(g3.containers[0], label_type='center')
    g3.set(ylabel=None)
    #plt.title("Sex")

    plt.subplot(1,3,3)
    g4 = sns.countplot(x='Site', data=df)
    g4.bar_label(g4.containers[0], label_type='center')
    g4.set(ylabel=None)
    #plt.title("Site")

    plt.suptitle(f"Demographic statistics for {set_tag} set (n={df.shape[0]})")

    fig.savefig(plot_path + f'demographics_{set_tag}_set.svg', bbox_inches='tight')

def plot_diagnosis_frequency(df: pd.DataFrame, set_tag: str):
    fig = plt.figure(1,(10,5))
    p = plt.bar(range(13), df.sum())
    plt.xticks(range(13), df.columns, rotation=90)
    plt.bar_label(p,label_type='center')
    plt.title(f"Diagnosis frequency for {set_tag} set (n={df.shape[0]})")

    fig.savefig(plot_path + f'diagnosis_frequency_{set_tag}_set.svg', bbox_inches='tight')

def plot_diagnosis_methods(df: pd.DataFrame, set_tag: str):
    dim = df.shape[1]
    cmap='viridis'

    IMPL = np.zeros((dim, dim))
    COOC = np.zeros((dim, dim))
    for i,c in enumerate(df):
        for j , cc in enumerate(df):
            IMPL[i,j] = 1-((1-df[cc])*df[c] ).mean()
            COOC[i,j] = ((df[cc])*df[c] ).mean()

    fig = plt.figure(1,(12,8))
    
    plt.subplot(2,2,1)
    sns.heatmap(IMPL,  xticklabels=[], yticklabels=df.columns, cmap=cmap)
    plt.title("Implication")
    
    plt.subplot(2,2,2)
    sns.heatmap(COOC, cmap=cmap, xticklabels=[], yticklabels=[])
    plt.title("Coocurrence")
    
    plt.subplot(2,2,3)
    sns.heatmap((COOC.T/COOC.diagonal()).T, xticklabels=df.columns, yticklabels=df.columns, cmap=cmap)
    plt.title("Association Rules")
    
    plt.subplot(2,2,4)
    sns.heatmap(df.corr(method='pearson'), xticklabels=df.columns, yticklabels=[], cmap=cmap)
    plt.title("Correlation")
    plt.suptitle(f"Diagnosis characterization for {set_tag} set (n={df.shape[0]})")

    fig.savefig(plot_path + f'diagnosis_characterization_{set_tag}_set.svg', bbox_inches='tight')

def plot_umap_cluster(df: pd.DataFrame, set_tag: str, ebd: any):
    fig = plt.figure(1,(15,14))

    for k in range(df.shape[1]):
        plt.subplot(4, 4, k+1)
        selec = df[df.columns[k]] != 0
        plt.scatter(ebd[~selec,0], ebd[~selec,1], s=2, c='lightgray', alpha=.1)
        plt.scatter(ebd[selec,0], ebd[selec,1], s=2, c='orange', alpha=.6)
        plt.title(df.columns[k], fontsize=8)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()

    fig.savefig(plot_path + f'umap_multi_cluster_{set_tag}_set.svg', bbox_inches='tight')