
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

plot_path = 'plots/'

########## Settings ###########

# Color management

color_palette = [
    "lightseagreen",
    "lightcoral",
    "powderblue",
    "steelblue",
    "hotpink",
    "seagreen"
]

#n_colors = 6
#discrete_cmap = cm.get_cmap('viridis', n_colors)
#color_palette = discrete_cmap(np.arange(n_colors))

# Convert CSS color strings to RGBA tuples
#css_colors = ["steelblue", "powderblue"]
#rgba_tuples = [mcolors.to_rgb(color) + (1,) for color in css_colors]

# Create custom colormap from RGBA tuples
#cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', rgba_tuples)

cmap = sns.color_palette("ch:start=.2,rot=-.3_r", as_cmap=True)
#cmap = 'viridis'

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
    g1 = sns.histplot(data=df, x='Age', bins=40, color=color_palette[3], edgecolor='white')
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
    
    case = 4 - df['Site'].nunique()
    
    plt.subplot(2,1,2)
    if case == 0:
        ax2 = sns.histplot(data=df, x='Diagnosis', hue="Site", palette=color_palette[2:], multiple='dodge', shrink=0.8, linewidth=0)
    elif case == 1:
        ax2 = sns.histplot(data=df, x='Diagnosis', hue="Site", palette=color_palette[2:-1], multiple='dodge', shrink=0.8, linewidth=0)
    else:
        pass

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

########## Statistical analysis ###########

def plot_global_scores(df: pd.DataFrame, metric: str):
    if metric == 'r2':   
        plt.figure(1,(6,2))
        sns.heatmap(df, cmap=cmap, fmt=".2f", vmin=df.min(axis=None), vmax=df.max(axis=None))
        plt.title(r"Mean $R^2$ scores for global features explaining confounders" + "\n")
        plt.savefig(plot_path + 'r2_features_confounder_global.svg', bbox_inches='tight')
    if metric == 'auroc':
        sns.heatmap(df, cmap=cmap, fmt=".2f", vmin=df.min(axis=None), vmax=df.max(axis=None))
        plt.title(r"Mean AUROC scores for global features explaining diagnoses" + "\n")
        plt.savefig(plot_path + 'auroc_features_diagnoses_global.svg', bbox_inches='tight')

def plot_subcortical_scores(score_dicts: dict, metric: str, measure_list, labels):
    
    global_vmin = +np.inf
    global_vmax = -np.inf

    for index in range(len(measure_list)):
        df = pd.DataFrame.from_dict(score_dicts[index], orient='index', columns=labels)
        if df.min(axis=None) < global_vmin:
            global_vmin = df.min(axis=None)
        if df.max(axis=None) > global_vmax:
            global_vmax = df.max(axis=None)

    if metric == 'r2':   
        plt.figure(1,(9,14))
        
        for index in range(len(measure_list)):
            df = pd.DataFrame.from_dict(score_dicts[index], orient='index', columns=labels)
            
            plt.subplot(7,1,index+1)
            ax = sns.heatmap(df, cmap=cmap, vmin=global_vmin, vmax=global_vmax, fmt=".2f")
            
            if index == len(measure_list)-1:
                ax.set_xticks(range(len(labels)), labels)
            else:
                ax.set_xticks([], [])
                ax.set(xlabel=None)
            
            plt.title(f"{measure_list[index]}")

        plt.suptitle(r"Mean $R^2$ scores for roi-based subcortical features (aseg) explaining confounders" + "\n")
        plt.tight_layout()
        plt.savefig(f'{plot_path}r2_features_confounder_aseg.svg', bbox_inches='tight')
    
    if metric == 'auroc':
        plt.figure(1,(12,20))

        for index in range(len(measure_list)):
            df = pd.DataFrame.from_dict(score_dicts[index], orient='index', columns=labels)
            
            plt.subplot(7,1,index+1)
            ax = sns.heatmap(df, cmap=cmap, yticklabels=df.index, vmin=global_vmin, vmax=global_vmax, fmt=".2f")
            
            if index == len(measure_list)-1:
                ax.set_xticks(range(len(labels)), labels)
            else:
                ax.set_xticks([], [])
                ax.set(xlabel=None)
            
            plt.title(f"{measure_list[index]}")

        plt.suptitle(r"Mean AUROC scores for roi-based subcortical features (aseg) explaining diagnoses" + "\n")
        plt.tight_layout()
        plt.savefig(f'{plot_path}auroc_features_diagnoses_aseg.svg', bbox_inches='tight')
    
def plot_cortical_scores(score_dicts: dict, metric: str, measure_list, labels, h_tag: str):
    
    global_vmin = +np.inf
    global_vmax = -np.inf

    for index in range(len(measure_list)):
        df = pd.DataFrame.from_dict(score_dicts[index], orient='index', columns=labels)
        if df.min(axis=None) < global_vmin:
            global_vmin = df.min(axis=None)
        if df.max(axis=None) > global_vmax:
            global_vmax = df.max(axis=None)

    if metric == 'r2':   
        plt.figure(1,(9,18))
        
        for index in range(len(measure_list)):
            df = pd.DataFrame.from_dict(score_dicts[index], orient='index', columns=labels)
            
            plt.subplot(9,1,index+1)
            ax = sns.heatmap(df, cmap=cmap, vmin=-0.015, vmax=0.008, fmt=".2f")
            
            if index == len(measure_list)-1:
                ax.set_xticks(range(len(labels)), labels)
            else:
                ax.set_xticks([], [])
                ax.set(xlabel=None)
            
            plt.title(f"{measure_list[index]}")

        plt.suptitle(r"Mean $R^2$ " + f"scores for roi-based cortical features (aparc {h_tag}) explaining confounders" + "\n")
        plt.tight_layout()
        plt.savefig(f'{plot_path}r2_features_confounder_aparc_{h_tag}.svg', bbox_inches='tight')
    
    if metric == 'auroc':
        plt.figure(1,(10,23))

        for index in range(len(measure_list)):
            df = pd.DataFrame.from_dict(score_dicts[index], orient='index', columns=labels)
            
            plt.subplot(9,1,index+1)
            ax = sns.heatmap(df, cmap=cmap, yticklabels=df.index, vmin=global_vmin, vmax=global_vmax, fmt=".2f")
            
            if index == len(measure_list)-1:
                ax.set_xticks(range(len(labels)), labels)
            else:
                ax.set_xticks([], [])
                ax.set(xlabel=None)
            
            plt.title(f"{measure_list[index]}")

        plt.suptitle(f"Mean AUROC scores for roi-based cortical features (aparc {h_tag}) explaining diagnoses" + "\n")
        plt.tight_layout()
        plt.savefig(f'{plot_path}auroc_features_diagnoses_aparc_{h_tag}.svg', bbox_inches='tight')

def plot_confounder_diagnoses_scores(df: pd.DataFrame):
    plt.figure(1,(5,2))
    sns.heatmap(df, cmap=cmap)
    plt.title(r"AUROC scores for confounders explaining diagnoses" + "\n")
    plt.savefig(plot_path + 'auroc_confounder_diagnoses_global.svg', bbox_inches='tight')

########## PR and ROC curves ###########
    
def plot_pr_curves(X_test: pd.DataFrame, Y_test: pd.DataFrame, lr_estimators: dict, hgb_estimators:dict, tag: str):
    fig, axs = plt.subplots(4,4, figsize=(20, 20))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

    counter = 0
    for i in range(0,4,1):
        for j in range(0,4,1):
            if (i > 2 & j > 0) or (counter > 12):
                fig.delaxes(axs[i,j])
            else:
                label = Y_test.columns[counter]
                counter += 1
                
                y_prob_lr = lr_estimators[label].predict_proba(X_test)[:, 1] 
                y_prob_hgb = hgb_estimators[label].predict_proba(X_test)[:, 1]
                
                PrecisionRecallDisplay.from_predictions(Y_test[label], y_prob_lr, pos_label=1, name="LR", ax=axs[i,j], color=color_palette[0])
                PrecisionRecallDisplay.from_predictions(Y_test[label], y_prob_hgb, pos_label=1, name="HGB", ax=axs[i,j], color=color_palette[1])
                
                axs[i,j].set_title(f"{label}")
                axs[i,j].legend(loc="best")
                axs[i,j].set_xlabel("Recall", size=12)
                axs[i,j].set_ylabel("Precision", size=12)
                axs[i,j].set_xmargin(0.01)
                axs[i,j].set_ymargin(0.01)
    
    fig.savefig(plot_path + f'pr_curves_{tag}.svg', bbox_inches='tight')

def plot_roc_curves(X_test: pd.DataFrame, Y_test: pd.DataFrame, lr_estimators: dict, hgb_estimators:dict, tag: str):
    fig, axs = plt.subplots(4,4, figsize=(20, 20))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

    counter = 0
    for i in range(0,4,1):
        for j in range(0,4,1):
            if (i > 2 & j > 0) or (counter > 12):
                fig.delaxes(axs[i,j])
            else:
                label = Y_test.columns[counter]
                counter += 1
                
                y_prob_lr = lr_estimators[label].predict_proba(X_test)[:, 1] 
                y_prob_hgb = hgb_estimators[label].predict_proba(X_test)[:, 1]
                
                RocCurveDisplay.from_predictions(Y_test[label], y_prob_lr, pos_label=1, name="LR", ax=axs[i,j], color=color_palette[0])
                RocCurveDisplay.from_predictions(Y_test[label], y_prob_hgb, pos_label=1, name="HGB", ax=axs[i,j], color=color_palette[1])
                
                axs[i,j].set_title(f"{label}")
                axs[i,j].legend(loc="best")
                axs[i,j].set_xlabel("False positive rate", size=12)
                axs[i,j].set_ylabel("True positive rate", size=12)
                axs[i,j].set_xmargin(0.01)
                axs[i,j].set_ymargin(0.01)
    
    fig.savefig(plot_path + f'roc_curves_{tag}.svg', bbox_inches='tight')