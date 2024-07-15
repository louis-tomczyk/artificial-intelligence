# ============================================================================#
# author        :   louis TOMCZYK
# goal          :   Definition of personalized AI functions
# ============================================================================#
# version       :   0.0.1 - 2021 09 29  - append_class
#                                       - display_circles
#                                       - display_factorial_planes
#                                       - display_parallel_coordinates
#                                       - display_parallel_coordinates_centroids
#                                       - display_scree_plot
#                                       - dot2png
#                                       - plot_dendrogram
#                                       - remove_several_variables
#                                       - remove_Data_Frame_variables
# ---------------
# version       :   0.1.0 - 2022 03 01  - Dcp
#                                       - Dcp_inv
#                                       - fill ground truth
#                                       - mapping_columns
#                                       - get_SC_index
#                                       - get_MC_index
#                                       - get_all_indexes
#                                       - distance_ground_truth
#                                       - get_ground_truth
#                                       - global_performance

# ============================================================================#

import numpy as np
import pandas
import pydot
import os
import seaborn as sns

from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.feature_extraction import DictVectorizer


# https://github.com/OpenClassrooms-Student-Center/Multivariate-Exploratory-Analysis/blob/master/functions.py
def append_class(df, class_name, feature, thresholds, names):
    '''Append a new class feature named 'class_name' based on a threshold split of 'feature'. 
    Threshold values are in 'thresholds' and class names are in 'names'.'''
    
    n = pd.cut(df[feature], bins = thresholds, labels=names)
    df[class_name] = n

            # ================================================#
            # ================================================#
            # ================================================#

# https://github.com/stenier-oc/realisez-une-analyse-de-donnees-exploratoire/blob/master/functions.py
def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(10,10))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

            # ================================================#
            # ================================================#
            # ================================================#

def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(10,10))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

            # ================================================#
            # ================================================#
            # ================================================#

def display_parallel_coordinates(df, num_clusters):
    '''Display a parallel coordinates plot for the clusters in df'''

    # Select data points for individual clusters
    cluster_points = []
    for i in range(num_clusters):
        cluster_points.append(df[df.cluster==i])
    
    # Create the plot
    fig = plt.figure(figsize=(12, 15))
    title = fig.suptitle("Parallel Coordinates Plot for the Clusters", fontsize=18)
    fig.subplots_adjust(top=0.95, wspace=0)

    # Display one plot for each cluster, with the lines for the main cluster appearing over the lines for the other clusters
    for i in range(num_clusters):    
        plt.subplot(num_clusters, 1, i+1)
        for j,c in enumerate(cluster_points): 
            if i!= j:
                pc = parallel_coordinates(c, 'cluster', color=[addAlpha(palette[j],0.2)])
        pc = parallel_coordinates(cluster_points[i], 'cluster', color=[addAlpha(palette[i],0.5)])

        # Stagger the axes
        ax=plt.gca()
        for tick in ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(20)        

            # ================================================#
            # ================================================#
            # ================================================#


def display_parallel_coordinates_centroids(df, num_clusters):
    '''Display a parallel coordinates plot for the centroids in df'''

    # Create the plot
    fig = plt.figure(figsize=(12, 5))
    title = fig.suptitle("Parallel Coordinates plot for the Centroids", fontsize=18)
    fig.subplots_adjust(top=0.9, wspace=0)

    # Draw the chart
    parallel_coordinates(df, 'cluster', color=palette)

    # Stagger the axes
    ax=plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20)    

            # ================================================#
            # ================================================#
            # ================================================#

                    
def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("Eigen value rank")
    plt.ylabel("Eigen value percentage")
    plt.title("Eigen Values Collapse")
    plt.show(block=False)

            # ================================================#
            # ================================================#
            # ================================================#

def dot2png(file_path_dot,file_name_png):
    (graph,)        = pydot.graph_from_dot_file(file_path_dot)
    graph.write_png(file_name_png)

            # ================================================#
            # ================================================#
            # ================================================#

palette = sns.color_palette("bright", 10)

            # ================================================#
            # ================================================#
            # ================================================#

                    
def plot_dendrogram(Z, names):
    plt.figure(figsize=(10,25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
    plt.show()

            # ================================================#
            # ================================================#
            # ================================================#


def remove_several_variables(Data_Frame, list_of_variables_to_remove):
    variables_0         = list(Data_Frame.axes[1])
    Indexes             = []

    for k in range(len(list_of_variables_to_remove)):
        Indexes.append(find_all_indexes(list_of_variables_to_remove[k],variables_0)[0][0])

    variables_1 = remove_using_index(Indexes,variables_0)
    
    return variables_1

            # ================================================#
            # ================================================#
            # ================================================#


def remove_Data_Frame_variables(Data_Frame, list_of_variables_to_remove):
    return Data_Frame[remove_several_variables(Data_Frame,list_of_variables_to_remove)]

    

# ----------------------
# Imprecision / Uncertainty functions
# ----------------------

            # ================================================#
            # ================================================#
            # ================================================#


# Cardinal and number of elements in the p-Meta Cluster
# -------
def Dcp(c,p):
    
    if p ==1:
        boundary_min    = 1;
        boundary_max    = c+1;
        Boundaries      = [boundary_min,boundary_max];        

    elif p == 2:
        boundary_min    = c+2;
        boundary_max    = c+1+binom(2,c);
        Boundaries      = [boundary_min,boundary_max];
    
    else:
        tmp_min         = [0 for k in range(1,p-2)];
        for q in range(1,p-2+1):
            tmp_min.append(binom(p-q,c))
            
        boundary_min    = c+2+np.sum(tmp_min);
        tmp_max         = [0 for k in range(1,p-1)];
        
        for q in range(p-2+1):
            tmp_max.append(binom(p-q,c))
            
        boundary_max = c+1+np.sum(tmp_max);
        Boundaries = [boundary_min,boundary_max];
        
    Card = Boundaries[1]-Boundaries[0]+1;
    
    if Boundaries[1]==Boundaries[0]:
        Boundaries = Boundaries[0]
        
    return Boundaries,Card
# -------

            # ================================================#
            # ================================================#
            # ================================================#

# get Meta cluster order from cluster index
# -------
def Dcp_inv(i,c):

    if i == 2**c:
        return c
    
    elif i <= c+1:
        return 1
    
    else:
        for pp in range(2,c+1):
            boundaries,card = Dcp(c,pp)
            
            if i >= boundaries[0] and i<= boundaries[1]:
                p = pp
                break

        return p
# -------

            # ================================================#
            # ================================================#
            # ================================================#

# fill the ground truth matrix with zeros
# -------
def fill_ground_truth(ground_truth):
    
    GT          = np.array(ground_truth.copy())
    n_rows,c    = GT.shape
    GT          = np.concatenate((np.zeros((n_rows,1)),GT,np.zeros((n_rows,2**c-c-1))),axis =1)

    return GT
# -------

            # ================================================#
            # ================================================#
            # ================================================#


# re-ordering the columns inside the p-MC
# -------
def mapping_columns(c):
    
    bins    = []
    n_bits  = np.log2(2**c)
    for k in range(2**c):
        bins.append(fill_binary(num2bin(k),n_bits))
    
    tmp = [[] for k in range(c+1)]
    for j in range(c+1):
        for k in range(len(bins)):
            if np.sum(bins[k]) == j:
                tmp[j].append(bins[k])
                
    tmp = flatten_list_of_list(tmp)
    
    return tmp
# -------

            # ================================================#
            # ================================================#
            # ================================================#


# get the index of the cluster to which the object belongs
# -------
def get_SC_index(ground_truth):
    return np.array(find_all_indexes(1,ground_truth)).astype(int)
# -------

            # ================================================#
            # ================================================#
            # ================================================#


# get the indexes of the columns that contain the cluster w_{tilde_i-1}
# -------
def get_MC_index(col_mapping_flattened,p,w):
    
    assert w!=0, "\n-------\n error\n-------\n\t cluster must be different of emptyset (w!=0) \n\t AND\n\t cluster must be inferior (or equal) to the meta cluster order asked (w<=p)"
    bins    = col_mapping_flattened   
    c       = np.log2(len(bins))
    
    assert p<=c, "\n-------\n error\n-------\n\t meta cluster order must be inferior to the size of the Frame of Discernment (p<=c)"
    tmp     = []
    
    for k in range(len(bins)):
        if (np.sum(bins[k])==p and bins[k][-w]==1):
            tmp.append(k)
            
    return list(1+np.array(tmp))
# -------

            # ================================================#
            # ================================================#
            # ================================================#


# get all the indexes of the mass values to change for the distance calculation
# -------
def get_all_indexes(ground_truth):

    GT          = fill_ground_truth(ground_truth)
    n_rows      = len(GT)
    n_cols      = len(GT[0])
    c           = int(np.log2(n_cols))
    indexes_SC  = np.zeros(n_rows)
    indexes_MC  = [[[] for j in range(c-1)] for k in range(n_rows)]
    tmp         = []
    all_indexes = [[] for k in range(n_rows)]
    
    for k in range(n_rows):
        indexes_SC[k]  = get_SC_index(GT[k])
        
        for p in range(2,c+1):
            GT_mapped = mapping_columns(c)
            indexes_MC[k][p-2] = get_MC_index(GT_mapped,p,int(indexes_SC[k]))
            
    for k in range(n_rows):
        tmp.append(flatten_list_of_list(indexes_MC[k]))

    for k in range(n_rows):
        all_indexes[k] =[int(indexes_SC[k]+1)]+tmp[k]
        
    return all_indexes
# -------

            # ================================================#
            # ================================================#
            # ================================================#

# Computation of the distance to the ground truth : VERSION 2
# -------
def distance_ground_truth(ground_truth,evidential_partition):

    GT              = fill_ground_truth(np.array(ground_truth))
    
#    print("\t filled ground truth =", GT)
    M               = np.array(evidential_partition).astype(float)
    all_indexes     = get_all_indexes(ground_truth)
    
    assert np.shape(GT)==np.shape(M), "\n ------\n error\n ------\n\t GT's and EP's shapes must correspond"
    
    n_rows,n_cols   = np.shape(M)
    c               = int(np.log2(n_cols))
    dist            = []
    
    for k in range(n_rows):
        M_tmp   = M[k][:]
        AI_tmp  = all_indexes[k][:]        
        dist_tmp= M_tmp
            
        for j in range(len(AI_tmp)):
            i = AI_tmp[j]
            
            # we look for the singleton clusters
            if j == 0:
                dist_tmp[i-1] = 1-M_tmp[i-1]
            else:
                p = Dcp_inv(i,c)
                dist_tmp[i-1] = M_tmp[i-1]/p
            
            d = np.sum(dist_tmp)
        dist.append(d)
        
    dist    = np.array(dist)/2
    d       = np.mean(dist)
    
    return dist,d
# -------

            # ================================================#
            # ================================================#
            # ================================================#

# Get Ground Truth matrix (assuming that highest value in credal partition blablabla)
# -------
def get_ground_truth(evidential_partition):
    
    M   = evidential_partition
    n_rows,n_cols = M.shape
    c   = int(np.log2(n_cols))
    GT  = np.zeros((n_rows,c))    

    Index_Max = M.argmax(axis=1)

    # case if the mass of belief is not in the singleton clusters    
    for k in range(n_rows):
        if Index_Max[k] == 0 or Index_Max[k] > c:
            Index_Max[k] = np.random.randint(1,c+1)
    
    for k in range(n_rows):
        GT[k,Index_Max[k]-1] = 1
    
    return GT
# -------

            # ================================================#
            # ================================================#
            # ================================================#

# Global performance
# -------
def global_performance(ground_truth,evidential_partition,SLU,SLI):
    
    M               = evidential_partition
    GT              = ground_truth
    n_rows,n_cols   = M.shape
    c               = int(np.log2(n_cols))
    
    Dist,d          = distance_ground_truth(GT,M)
    Dcp_mat         = np.zeros((c,2));
    eta             = np.zeros(n_rows);
    
    M_prime         = np.zeros((n_rows,c));
    I_prime         = np.zeros((n_rows,c));
    U_prime         = np.zeros((n_rows,c));
    
    for p in range(1,c+1):
        Dcp_mat[p-1,:],card = Dcp(c,p);

    Dcp_mat         = np.array(Dcp_mat).astype(int)
    
    for k in range(n_rows):
        for p in range(c):

            M_prime[k,p]    = np.sum(M[k,Dcp_mat[p,0]-1:Dcp_mat[p,1]]);
            I_prime[k,p]    = np.sum(SLI[k,Dcp_mat[p,0]-1:Dcp_mat[p,1]]);
            U_prime[k,p]    = np.sum(SLU[k,Dcp_mat[p,0]-1:Dcp_mat[p,1]]);
                  
        eta[k] = (1-Dist[k])*(4/3-1/3*np.sum(M_prime[k,:]*(1+I_prime[k,:])*(1+U_prime[k,:])));
    eta_mean = np.mean(eta)

    return Dist,eta_mean
# -------
    
