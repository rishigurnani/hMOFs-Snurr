import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = "Arial Narrow"
rcParams['font.family'] = "sans-serif"


###############################################################################
# Change here to use different property
property_X = 'DFT_bandgap'
property_Y = 'exp_Tg'
property_C = 'DFT_dielectric_total'
property_S = 'exp_rho'   # set '' for no size variation
###############################################################################





data_set_label = '11641'
marker = 'o'
colormap = 'viridis'
figure_out_dir = 'plot/'


source = dict()
source['DFT_bandgap'] = {'property_name':'Bandgap, bulk (eV)', 'lim':[0.2,9.8], 'scale':'linear'}
source['DFT_dielectric_total'] = {'property_name':'Dielectric constant', 'lim':[1.5,8.5], 'scale':'linear'}
source['exp_Tg'] = {'property_name':'Glass transition temp. (K)', 'lim':[10,790], 'scale':'linear'}
source['exp_rho'] = {'property_name':'Density (g/cc)', 'lim':[0.5,2.3], 'scale':'linear'}


#with plt.xkcd():
    



###############################################################################
# X, Y, color, size setting
# X values
file_X = 'data/predict_' + data_set_label + '_' + property_X + '.csv'
df_X = pd.read_csv(file_X)
X = df_X['y']

# Y values
file_Y = 'data/predict_' + data_set_label + '_' + property_Y + '.csv'
df_Y = pd.read_csv(file_Y)
Y = df_Y['y']

# Color values
file_C = 'data/predict_' + data_set_label + '_' + property_C + '.csv'
df_C = pd.read_csv(file_C)
C = df_C['y']
print('color: ', data_set_label, property_C, min(C), ' ~ ',  max(C))

C = (C - min(C)) / (max(C)-min(C))


# Size values
if property_S != "":
    file_S = 'data/predict_' + data_set_label + '_' + property_S + '.csv'
    df_S = pd.read_csv(file_S)
    S = df_S['y']
    print('Size: ', data_set_label, property_S, min(S), ' ~ ',  max(S))
    S = S.rank()
    S = (S - min(S)) / (max(S)-min(S))
    S = S + 1
    S = (S*5) **5  / 160 / 2 /2 / 2 /2 + 20           
else:
    S = 20
###############################################################################


label_X = source[property_X]['property_name'] 
label_Y = source[property_Y]['property_name']

lim_X = source[property_X]['lim']
lim_Y = source[property_Y]['lim']

scale_X = source[property_X]['scale']
scale_Y = source[property_X]['scale']


###############################################################################
# Plot start

fig = plt.figure(figsize=(7,7))

#ax = plt.axes([0.14, 0.14, 0.72, 0.72])

plt.scatter(X, Y, marker=marker, c=C, linewidths=0, s=S, alpha=1, cmap=plt.get_cmap(colormap), zorder=10)


plt.xlabel(label_X,size = 22)
plt.ylabel(label_Y,size = 22)

plt.xlim(lim_X)
plt.ylim(lim_Y)

plt.xscale(scale_X)
plt.yscale(scale_Y)


#plt.axis('off')
plt.tick_params(axis= 'both', which= 'major', direction= 'in', length=8, labelsize=13)
plt.tick_params(axis= 'both', which= 'minor', direction= 'in', length=3)
#    ax.tick_params(which="minor", axis="x", direction="in")

filename = figure_out_dir + property_X + '_' + property_Y +  '.png'

plt.tight_layout()
#plt.show()
plt.savefig(filename,  dpi=450, transparent=True)




