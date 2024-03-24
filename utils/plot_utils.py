
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import generate_undersampled_set, generate_oversampled_set, generate_label_stats
from sklearn.model_selection import train_test_split
from umap import UMAP

plot_path = 'plots/'

########## Settings ###########

color_palette = [
    "lightseagreen",
    "lightcoral",
    "powderblue",
    "steelblue",
    "hotpink",
    "seagreen"
]

diagnosis_acronyms = ['TSD', 'DD', 'ADHD', 'MD', 'ASD', 'CD', 'OD', 'SLD', 'OCD', 'D', 'ID', 'ED', 'AD', 'NC']

hue_order = ['Female', 'Male']

########## Helper methods ###########

def aggregate_diagnoses(C: pd.DataFrame, D: pd.DataFrame) -> pd.DataFrame:
    df_tmp = pd.concat([C.iloc[:,:3], D], axis=1)
    
    # Diagnoses
    id_vars = list(df_tmp.iloc[:,:3].columns)
    value_vars = df_tmp.iloc[:,3:].columns
    df_1 = df_tmp.melt(id_vars=id_vars, value_vars=value_vars, var_name='Diagnosis', value_name='value')
    df_1 = df_1[df_1['value'] != 0]
    df_1 = df_1.drop(columns='value')

    # Controls
    df_2 = df_tmp[D.eq(0).all(axis=1)]
    df_2 = df_2.iloc[:,:3]
    df_2['Diagnosis'] = 'No_Condition'

    data = pd.concat([df_1, df_2], axis=0)
    return data

########## Demographics ###########

def plot_demographics(df: pd.DataFrame, tag: str):
    fig = plt.figure(1,(14,4))

    plt.subplot(1,3,1)
    g1 = sns.histplot(data=df, x='Age', bins=40, color=color_palette[3])
    g1.set(ylabel=None)
    g1.set_yticklabels([])
    g1.set_yticks([])

    plt.subplot(1,3,2)
    g3 = sns.histplot(data=df, x='Site', hue='Sex', hue_order=hue_order, palette=color_palette[0:2], multiple='dodge', shrink=0.8, linewidth=0)
    g3.bar_label(g3.containers[0], label_type='center')
    g3.bar_label(g3.containers[1], label_type='center')

    plt.subplot(1,3,3)
    sns.violinplot(data=df, x='Site', y='Age', hue='Sex', palette=color_palette[0:2], density_norm='count', inner='quart', legend=None, fill=False, split=True, gap=0.2)

    plt.suptitle(f"Demographic statistics for {tag} (n={df.shape[0]})")
    fig.savefig(plot_path + f'demographics_{tag}.svg', bbox_inches='tight')

########## Diagnoses ###########
    
def plot_diagnosis_frequency(df: pd.DataFrame, tag: str):
    fig = plt.figure(1,(17,4))
    ax = sns.violinplot(data=df, x='Diagnosis', y='Age', hue="Sex", hue_order=hue_order, palette=color_palette[0:2], density_norm='width', inner='quart', legend=None, fill=False, split=True, gap=0.2)
    ax.set_axisbelow(True)
    #ax.grid(which='both', color='gray', alpha=0.1)
    ax.set_xticks(range(14), diagnosis_acronyms)
    ax.set(xlabel=None)
    plt.title("Diagnosis frequency for age and sex")

    fig.savefig(plot_path + f'diagnosis_frequency_{tag}.svg', bbox_inches='tight')

def plot_diagnosis_prevalence(df: pd.DataFrame, tag: str):
    fig = plt.figure(1,(17,8))
    
    plt.subplot(2,1,1)
    ax1 = sns.histplot(data=df, x='Diagnosis', hue="Sex", hue_order=hue_order, palette=color_palette[0:2], multiple='dodge', shrink=0.8, linewidth=0)
    ax1.set_axisbelow(True)
    #ax1.grid(which='both', color='gray', alpha=0.05)
    ax1.set_yticks(range(0,1200,200))
    ax1.set_xticks(range(14), [])
    ax1.set(xlabel=None)
    #plt.title("Diagnosis distribution for sex and site")
    
    offset = 4-df['Site'].nunique()
    
    plt.subplot(2,1,2)
    ax2 = sns.histplot(data=df, x='Diagnosis', hue="Site", palette=color_palette[2+offset:], multiple='dodge', shrink=0.8, linewidth=0)
    #ax2.grid(which='both', color='gray', alpha=0.05)
    #ax2.set_xticks(range(14), data['Diagnosis'].unique().tolist(), rotation=90)
    ax2.set_xticks(range(14), diagnosis_acronyms)
    ax2.set(xlabel=None)
    #plt.title("Diagnosis distribution for site")
   
    plt.suptitle(f"Diagnosis prevalence for sex and site")
    plt.subplots_adjust(left=None, bottom=None, right=None, top=0.95, wspace=None, hspace=0.05)

    fig.savefig(plot_path + f'diagnosis_prevalence_{tag}.svg', bbox_inches='tight')

def plot_diagnosis_frequency_simple(df: pd.DataFrame, tag: str):
    fig = plt.figure(1,(10,5))
    p = plt.bar(range(13), df.sum())
    plt.xticks(range(13), df.columns, rotation=90)
    plt.bar_label(p,label_type='center')
    plt.title(f"Diagnosis frequency for {tag} set (n={df.shape[0]})")

    fig.savefig(plot_path + f'diagnosis_frequency_simple_{tag}.svg', bbox_inches='tight')

def plot_diagnosis_methods(df: pd.DataFrame, tag: str):
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
    plt.suptitle(f"Diagnosis characterization for {tag} (n={df.shape[0]})")

    fig.savefig(plot_path + f'diagnosis_characterization_{tag}.svg', bbox_inches='tight')

########## UMAP ###########

def plot_umap(df: pd.DataFrame, tag: str):
    umap = UMAP()
    ebd = umap.fit_transform(df)
    
    fig = plt.figure(1,(5,4))
    plt.scatter(ebd[:,0], ebd[:,1], s=2, color='lightgrey')
    plt.title('Scatter plot in embedding space')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()

    fig.savefig(plot_path + f'umap_{tag}.svg', bbox_inches='tight')   

def plot_umap_combined(X: pd.DataFrame, Y: pd.DataFrame, tag: str):
    # Create new sets for oversampling and undersampling
    X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.25, random_state=0)
    
    # regular
    _, mean_ir_regular = generate_label_stats(Y, mean_ir=True)

    # under
    _, Y_under = generate_undersampled_set(X_train, Y_train)
    _, mean_ir_under = generate_label_stats(Y_under, True)
    
    # over
    _, Y_over = generate_oversampled_set(X_train, Y_train)
    _, mean_ir_over = generate_label_stats(Y_over, True)

    # Fit umap
    umap = UMAP()
    ebd_1 = umap.fit_transform(Y)
    ebd_2 = umap.fit_transform(Y_under)
    ebd_3 = umap.fit_transform(Y_over)

    # Plot
    fig = plt.figure(1,(11,4))

    plt.subplot(1, 3, 1)
    ax1 = sns.scatterplot(x=ebd_1[:,0], y=ebd_1[:,1], s=2, color=color_palette[3])
    ax1.set_xticks([], [])
    ax1.set_yticks([], [])
    plt.title(f"Regular (n={Y.shape[0]}, mean-ir={mean_ir_regular:.1f})")

    plt.subplot(1, 3, 2)
    ax2 = sns.scatterplot(x=ebd_2[:,0], y=ebd_2[:,1], s=2, color=color_palette[3])
    ax2.set_xticks([], [])
    ax2.set_yticks([], [])
    plt.title(f"Undersampled (n={Y_under.shape[0]}, mean-ir={mean_ir_under:.1f})")

    plt.subplot(1, 3, 3)
    ax3 = sns.scatterplot(x=ebd_3[:,0], y=ebd_3[:,1], s=2, color=color_palette[3])
    ax3.set_xticks([], [])
    ax3.set_yticks([], [])
    plt.title(f"Oversampled (n={Y_over.shape[0]}, mean-ir={mean_ir_over:.1f})")
    
    plt.suptitle(f"Scatter plots for diagnoses in embedding space")
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    
    plt.tight_layout()
    fig.savefig(plot_path + f'umap_combined_{tag}.svg', bbox_inches='tight')   


def plot_umap_cluster(df: pd.DataFrame, tag: str):
    umap = UMAP()
    ebd = umap.fit_transform(df)
    
    fig = plt.figure(1,(15,14))
    for k in range(df.shape[1]):
        plt.subplot(4, 4, k+1)
        selec = df[df.columns[k]] != 0
        plt.scatter(ebd[~selec,0], ebd[~selec,1], s=2, c='lightgray', alpha=.1)
        plt.scatter(ebd[selec,0], ebd[selec,1], s=2, c=color_palette[3], alpha=.6)
        plt.title(df.columns[k], fontsize=8)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()

    fig.savefig(plot_path + f'umap_multi_cluster_{tag}.svg', bbox_inches='tight')

########## Morphometric features ###########
    
def plot_feature_confounder_relation(df: pd.DataFrame, f: str, c:str, c2:str, h_list: list[str]):
    
    fig = plt.figure(1,(11,4))
    
    plt.subplot(1, 2, 1)
    ax1 = sns.scatterplot(data=df[df[c2] == h_list[0]], x=c, y=f, s=2, color=color_palette[1])
    ax1.set_ylim(0.5, 2.3)
    plt.title(f"{h_list[0]}")

    plt.subplot(1, 2, 2)
    ax2 = sns.scatterplot(data=df[df[c2] == h_list[1]], x=c, y=f, s=2, color=color_palette[0])
    ax2.set_ylim(0.5, 2.3)
    plt.title(f"{h_list[1]}")

    plt.suptitle(f"Relationship between {f} and {c}")
    fig.savefig(plot_path + f'relationship_{f}_{c}_with_{c2}.svg', bbox_inches='tight')