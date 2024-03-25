import pandas as pd
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

########## UMAP ###########

def plot_umap(df: pd.DataFrame, tag: str):
    umap = UMAP(random_state=33)
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
    umap = UMAP(random_state=33)
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
    
    plt.suptitle(f"Scatter plots for labels in embedding space")
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    
    plt.tight_layout()
    fig.savefig(plot_path + f'umap_combined_{tag}.svg', bbox_inches='tight')   


def plot_umap_cluster(df: pd.DataFrame, tag: str):
    umap = UMAP(random_state=33)
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