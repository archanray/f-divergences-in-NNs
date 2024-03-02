import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import seaborn as sns
import os
from tsnecuda import TSNE

def plot_data(data, labels, savedir="plots", dataset="mnist", iteration=0):
    # need to handle data and labels into cpu later

    # the number of components = dimension where data is being projected
    model = TSNE(n_components=2, random_state=0)

    tsne_data = model.fit_transform(data)
    
    # creating a new data frame which help us in ploting the result data
    tsne_data = np.vstack((tsne_data.T, labels)).T
    tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
    
    # Ploting the result of tsne
    sns.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, "Dim_1", "Dim_2").add_legend()

    savepath = os.path.join(savedir, dataset)
    if not os.path.isdir(savedir):
        os.makedirs(savepath)
    plt.savefig(os.path.join(savepath, str(iteration)+"TSNE_plot.py"))
    
    return None


def plot_data_cuda(data, labels, savedir="plots", dataset="mnist", iteration=0):
    # need to handle data and labels into cpu later

    tsne = TSNE(n_iter=1000, verbose=1, num_neighbors=64)
    tsne_results = tsne.fit_transform(data.reshape(60000,-1))

    # Create the figure
    fig = plt.figure( figsize=(8,8) )
    ax = fig.add_subplot(1, 1, 1, title='TSNE' )

    # Create the scatter
    ax.scatter(x=data[:,0], y=data[:,1], c=labels, cmap=plt.cm.get_cmap('Paired'),
                    alpha=0.4, s=0.5)
    
    savepath = os.path.join(savedir, dataset)
    if not os.path.isdir(savedir):
        os.makedirs(savepath)
    plt.savefig(os.path.join(savepath, str(iteration)+"TSNE_plot.py"))
    return None