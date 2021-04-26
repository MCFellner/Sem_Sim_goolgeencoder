import numpy as np
import pandas as pd
import scipy as sp
import sys
import seaborn as sns
import matplotlib.pyplot as plt
#import d00_utils.fileutils as fileutils
import os
import tensorflow_hub as hub
import tensorflow_text


# function collection

def read_embed_data(csv_file,column2embed,columns2keep,sel_embed='multi',column2dropna=[]):
    """function reading in item based csv with namings, 
    returns pandas dataframe with specified columns plus embeddings
    Parameters: 
    csv_file (str): csv file with encoding='UTF-8' (for umlauts) 
    column2embed (str): column with names/sentences to get embeddings
    columns2keep (list/str): select columns to keep from csv (important: need to be unique to not cause merging errors)
    column2dropna (list/str), optional: column with nan that defined named trials
    Returns: 
    int: Description of return value 
    """
    #load csv with data
    data=pd.read_csv(csv_file,encoding='UTF-8')
    data=data[columns2keep]
    
    #get sentence encoder
    if sel_embed=='multi':
        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
    elif sel_embed=='eng':
        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    else:
        raise ValueError('sel embed not defined')
        
    # only select named pics
    named_data=data.dropna(subset=[column2dropna]).reset_index(drop=True)   
    #get word embeddings
    embed_names=embed(named_data[column2embed].values.tolist())
    embed_values=embed_names.numpy()    
    # add embeddings to dataframe
    dim_names=['dim'+str(x+1) for x in range(embed_values.shape[1])]
    embed_df=pd.DataFrame(embed_values,columns=dim_names)
    embed_df=pd.concat([named_data, embed_df], axis=1)   
    #merge with data 
    data=data.merge(embed_df, on=columns2keep, how='left')
    return data


def plot_similarity(labels, features, rotation, save_option=False, fig_file=[]):
  """ plot similarity matrix for sentence embeddings 
  
  Parameters:
      labels (list): embedded sentences
      features (np.array): embeddings of sentences
      rotation (int): rotation of labels
      save_option (bool): save fig
      fig_file (str): filename to save figure to"""
  corr = np.inner(features, features)
  sns.set(font_scale=1.2)
  g = sns.heatmap(
      corr,
      xticklabels=labels,
      yticklabels=labels,
      vmin=0,
      vmax=1,
      cmap="YlOrRd")
  g.set_xticklabels(labels, rotation=rotation)
  g.set_title("Semantic Textual Similarity")
  if save_option:
      plt.savefig(fig_file, bbox_inches='tight')
  plt.close('all')

def plot_for_each_type(data,feature_columns,sel_col,column2embed,column2dropna,sel_path):
    all_items=pd.unique(data[sel_col]).tolist()    
    for sel_item in all_items:
        sel_data=data[data[sel_col]==sel_item].dropna(subset=[column2dropna])
        filename=os.path.join(sel_path,'figure_'+str(sel_item).replace('.','_')+'.svg')
        plot_similarity(sel_data[column2embed].tolist(), sel_data[feature_columns],90,save_option=True,fig_file=filename)


def create_folder(*args):
    import os
    sel_path=os.path.join(*args)
    if not os.path.exists(sel_path):
        os.makedirs(sel_path)
    return  sel_path

# plot embeddings similarity matrizes/ save as svg

# quantify similarity for each pic

# write embeddings to file (merge pandas df)

# pcas on dimensions


########### pipeline
# %% plot embeddings similarity matrizes/ save as svg
all_data=pd.DataFrame()
all_paradigms=['pilot','mem']  #,'fmri']
project_path='D:\\squiggles\\scripts\\behav\\sem_sim_py'

for paradigm in all_paradigms:
    csv_file=os.path.join(project_path,'data',paradigm+'_namings.csv')
    
    
    column2embed='corrected'
    columns2keep=['corrected','subject','Pic','named','PicID']
    column2dropna='named' #leave empty if not needed
    
    data=read_embed_data(csv_file,column2embed,columns2keep,'multi',column2dropna)
    
    feature_columns=[x for x in list(data.columns) if 'dim' in x]
    
    #plot similarity maps for each subject
    sel_path=create_folder(project_path,'output','figures',paradigm,'semsim','subject')
    plot_for_each_type(data,feature_columns,'subject',column2embed,column2dropna,sel_path)
    
    # plot similarity map for each item
    sel_path=create_folder(project_path,'output','figures',paradigm,'semsim','item')
    plot_for_each_type(data,feature_columns,'Pic',column2embed,column2dropna,sel_path)
    
    
    # write csv file output
    sel_path=create_folder(project_path,'data')
    filename=os.path.join(sel_path,paradigm+'multiling_embed.csv')

    data.to_csv(filename, index=False)
    data['paradigm']=paradigm
    # combine all paradigms
    all_data=all_data.append(data)

# %% compare similarities for english embeddings

all_data_en=pd.DataFrame()
all_paradigms=['pilot','mem']  #,'fmri']
for paradigm in all_paradigms:
    csv_file=os.path.join(project_path,'data',paradigm+'_namings.csv')
       
    column2embed='EnglischGoogleEncoder'
    columns2keep=['EnglischGoogleEncoder','corrected','subject','Pic','named','PicID']
    column2dropna='named' #leave empty if not needed
    
    data=read_embed_data(csv_file,column2embed,columns2keep,'eng',column2dropna)
    
    feature_columns=[x for x in list(data.columns) if 'dim' in x]
    
    #plot similarity maps for each subject
    sel_path=create_folder(project_path,'output','figures',paradigm,'semsim_eng','subject')
    plot_for_each_type(data,feature_columns,'subject',column2embed,column2dropna,sel_path)
    
    # plot similarity map for each item
    sel_path=create_folder(project_path,'output','figures',paradigm,'semsim_eng','item')
    plot_for_each_type(data,feature_columns,'Pic',column2embed,column2dropna,sel_path)
    
    
    # write csv file output
    sel_path=create_folder(project_path,'data')
    filename=os.path.join(sel_path,paradigm+'eng_embed.csv')

    data.to_csv(filename, index=False)
    data['paradigm']=paradigm
    # combine all paradigms
    all_data_en=all_data_en.append(data)

# %% get correlation between mulitlanguage german embeddings and english embeddings
feature_columns=[x for x in list(all_data.columns) if 'dim' in x]
# only select named trials
sel_data=all_data.dropna(subset=[column2dropna]).reset_index(drop=True) 
sel_data_en=all_data_en.dropna(subset=[column2dropna]).reset_index(drop=True) 

sim_mat_multi= np.inner(sel_data[feature_columns], sel_data[feature_columns])
sim_mat_eng= np.inner(sel_data_en[feature_columns], sel_data_en[feature_columns])

rho_multi2eng=sp.stats.spearmanr(sim_mat_multi.flatten(),sim_mat_eng.flatten())

diff_sim=sim_mat_multi-sim_mat_eng

# find items highest average abs difference
mean_simdiff_peritem=np.mean(np.abs(diff_sim),axis=0)

# select items with highes difference and check the namings
perc95=np.percentile(mean_simdiff_peritem,95)

high_diff_data=sel_data_en.iloc[mean_simdiff_peritem>perc95,:]
# %% get similarity per item

# loop through all unique Pic
all_items=pd.unique(all_data['Pic']).tolist()    
named_data=all_data.dropna(subset=[column2dropna])

simvalue_item=np.empty(len(all_items),dtype=object)
count_named=np.empty_like(simvalue_item)
count_shown=np.empty_like(simvalue_item)

for i, sel_item in enumerate(all_items):
    sel_data=named_data[named_data['Pic']==sel_item]
    count_named[i]=len(named_data[named_data['Pic']==sel_item])
    count_shown[i]=len(all_data[all_data['Pic']==sel_item])
    # get embeddings values & calculate inner
    item_sim = np.inner(sel_data[feature_columns], sel_data[feature_columns])
    sel_ind=np.triu_indices(len(item_sim), k=+1, m=None)
    simvalue_item[i]=np.mean(item_sim[sel_ind])
    
    
# perm stats distribution of within pic similarit
n_rand=1000
simvalue_itemrand=np.empty((len(all_items),n_rand),dtype=object)
for r in range(n_rand):
    # randomize Pic column
    rand_pic=np.random.permutation(named_data.loc[:,'Pic'])
    for i, sel_item in enumerate(all_items):
        sel_data=named_data[rand_pic==sel_item]
        # get embeddings values & calculate inner
        item_sim = np.inner(sel_data[feature_columns], sel_data[feature_columns])
        sel_ind=np.triu_indices(len(item_sim), k=+1, m=None)
        simvalue_itemrand[i,r]=np.mean(item_sim[sel_ind])


# get rank for each pic
sorted_simvalue=np.sort(simvalue_itemrand)
ps_pic=np.sum(np.greater_equal(sorted_simvalue,simvalue_item[:,np.newaxis]),axis=-1)/n_rand;

#get mean distribution
sorted_meansimvalue=np.sort(np.nanmean(simvalue_itemrand,axis=0))
mean_acrosspics=np.nanmean(simvalue_item,axis=0)
p_all=np.sum(np.greater_equal(sorted_meansimvalue[np.newaxis,:],mean_acrosspics),axis=-1)/n_rand;

# save info with pic metrics (sim value for each each pic,p-value for each pic, count names, count shown)

simvalue_pic=pd.DataFrame({'Pic':all_items,'simvalue':simvalue_item,'p_value_perm':ps_pic,'count_named':count_named,'count_shown':count_shown})

# write csv file output
sel_path=create_folder(project_path,'data')
filename=os.path.join(sel_path,'multiling_embed_picinfo_eeg.csv')
data.to_csv(filename, index=False)

# plot hist with simvalues across pics
g=sns.histplot(data=simvalue_pic,x='simvalue', stat='probability')
g.axvline(mean_acrosspics,color='k',linestyle='--')
g.axvline(np.nanmean(sorted_meansimvalue),color='grey',linestyle='--')

g.set_xlabel('mean similarity in naming')
g.set_ylabel('proportion of items')
g.set_title('perm stat p= '+str(p_all))
fig=g.get_figure()
fig.savefig('D:\\squiggles\\scripts\\behav\\sem_sim_py\\output\\figures\\mean_sim_eachitem_eeg.svg', bbox_inches='tight')

# check which dimension explain most variance in semsim (only use )


# %% pca (also dimension reduction for clustering)

from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
 
X=named_data[feature_columns].to_numpy()
X=StandardScaler().fit_transform(X)
pca=decomposition.PCA(whiten=True)

pca.fit(X)

plt.plot(pca.explained_variance_ratio_.cumsum())

# dim reduction: only take pc up to 90% variance
num_pc=np.sum(pca.explained_variance_ratio_.cumsum()<=0.9)
X_trans=pca.transform(X)
X_red=X_trans[:,:num_pc]

# k means clustering 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
sil=[]
kmax=50
for k in range(2,kmax+1):
    kmeans=KMeans(n_clusters=k).fit(X)
    labels=kmeans.labels_
    sil.append(silhouette_score(X,labels,metric='cosine'))
plt.plot(sil)    
# no real peak sil value, k means not effective

# %% hierarchical clustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

X=named_data[feature_columns].to_numpy()

dissimilarities=1-np.inner(X,X)
dissimilarities[dissimilarities<0]=0

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None )

model = model.fit(dissimilarities)
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level')
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()


# %% spectral clustering

from sklearn.cluster import SpectralClustering
X=named_data[feature_columns].to_numpy()

similarities=np.inner(X,X)

specclus=SpectralClustering(n_clusters=30,affinity='precomputed',assign_labels='discretize')
specclus=specclus.fit(similarities)

labels=specclus.labels_

# %% mds
from sklearn.manifold import MDS

# calculate similarities using np.inner

dissimilarities=1-np.inner(X,X)
dissimilarities[dissimilarities<0]=0
mds=MDS(dissimilarity='precomputed')
mds=mds.fit(dissimilarities)
pos=mds.embedding_

plt.figure(figsize=[60,40])
ax=plt.scatter(pos[:,0],pos[:,1],alpha=0.2)
# add text to every 30 data point
pic_ind=np.linspace(0,len(X_red)-1,num=300,dtype=int)

for i in range(len(X_red)):
    ax=plt.text(pos[i,0],pos[i,1],named_data.corrected.iloc[i],fontsize='xx-small')

fig=ax.get_figure()
fig.savefig('D:\\squiggles\\scripts\\behav\\sem_sim_py\\output\\figures\\mds_test_innersim.svg', bbox_inches='tight')


# %% tsne
from sklearn.manifold import TSNE

#tsne on similarity matrix
tsne=TSNE(metric='precomputed',)
tsne.fit(dissimilarities)

pos_tsne=tsne.embedding_

plt.figure(figsize=[60,40])
ax=plt.scatter(pos_tsne[:,0],pos_tsne[:,1],alpha=0.2)
# add text to every 30 data point
pic_ind=np.linspace(0,len(X_red)-1,num=300,dtype=int)

for i in range(len(X_red)):
    ax=plt.text(pos_tsne[i,0],pos_tsne[i,1],named_data.corrected.iloc[i],fontsize='xx-small')

fig=ax.get_figure()
fig.savefig('D:\\squiggles\\scripts\\behav\\sem_sim_py\\output\\figures\\tsne_test_innersim.svg', bbox_inches='tight')
