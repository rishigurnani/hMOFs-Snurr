import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from matplotlib import rcParams
rcParams['font.sans-serif'] = "Arial Narrow"
rcParams['font.family'] = "sans-serif"
import sys

args = sys.argv

path = args[1]
fp_key_for_coloring = args[2]
start_str = args[3]
end_str = args[4]
n = int(args[5]) #number of points to be plotted on top 

fp_df = pd.read_csv(path, index_col=0)

for ind,col in enumerate(fp_df.columns):
    if start_str in col:
        start_col = ind + 1
    elif end_str in col:
        end_col = ind

X_heads = list(fp_df.columns[start_col:end_col]) + ['Dom._Pore_(ang.)',
'Max._Pore_(ang.)',
'Void_Fraction',
'Surf._Area_(m2/g)',
'Vol._Surf._Area',
'Density']

try:
    C = fp_df[fp_key_for_coloring]
except:
    pass

def pcaDF(fp_df, X_heads):
    '''
    This function returns the pcaDF of fp_df
    '''
    fp_df = fp_df.drop_duplicates(keep='first') # delete duplication
    fp_df = fp_df.fillna(0) # fill zero in NA columns

    X_original = fp_df[X_heads]
    X_length = len(X_heads)

    # For PCA,  scaling must be done!
    # Otherwise, larger value containing X component will always be collected as important PC
    xscale = preprocessing.MinMaxScaler()
    X = xscale.fit_transform(X_original)

    n_data = len(X)
    ###############################################################



    print('Original fingerprint')  
    print('    Number of dataset =', n_data)
    print('    Dimension of X    =', X_length)



       ###############################################################
    # PCA
    n_components = min(X_length, n_data)   # reset to max size of X
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    PC_heads = list()
    for i in range(len(X[0])):
        PC_heads.append('PC' + str(i))

    X_pca_df = pd.DataFrame(data = X_pca, columns = PC_heads)
    return X_pca_df

def getTopUptakes(fp_df, n):
    top_inds = fp_df['CH4_v/v_1_bar'].sort_values()[-n:].index
    return fp_df.ix[top_inds]

X_pca_df = pcaDF(fp_df, X_heads)

top_df = getTopUptakes(fp_df, n)

X_pca_top = pcaDF(top_df, X_heads)


PC1 = X_pca_df['PC0']
PC2 = X_pca_df['PC1']


#new
PC1_top = X_pca_top['PC0']
PC2_top = X_pca_top['PC1']


pca_map = plt.scatter(PC1, PC2, 
            marker='o', 
            c=C, 
            linewidths=0, 
            s=20, 
            alpha=1, 
            cmap=plt.get_cmap('cool'))

f, ax = plt.subplots(1)

ax.scatter(PC1, PC2, 
            marker='o', 
            c=C, 
            linewidths=0, 
            s=20, 
            alpha=1, 
            cmap=plt.get_cmap('cool'))
ax.scatter(PC1_top, PC2_top, 
            marker='o', 
            c='fuchsia', 
            linewidths=0, 
            s=20, 
            alpha=1)

plt.xlabel('PC1')
plt.ylabel('PC2')

if fp_key_for_coloring != '':
    #cax3 = f.add_axes([0.7, 0.22, 0.25, 0.015])
    #cbar = plt.colorbar(pca_map, cax=cax3, orientation='horizontal', pad=.2)#, ticks=ticks)
    cbar = plt.colorbar(pca_map, orientation='horizontal', pad=.2)#, ticks=ticks)
    cbar.outline.set_linewidth(0)
    cbar.set_label(fp_key_for_coloring, fontsize=14)
    cbar.ax.tick_params(labelsize=12)#, width=0)
    cbar.ax.set_xlabel('CH4 Uptake at 1 bar')

plt.tick_params(axis= 'both', which= 'both', direction= 'in')

f.set_size_inches(12, 5)
#plt.tight_layout()
plt.show()
plt.savefig('./PCA.png', dpi=450)
plt.close()







