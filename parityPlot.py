import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
import rishi_utils as ru


# # Data source
# df_train = pd.read_csv('data/train.csv')
# df_test = pd.read_csv('data/test.csv')


# #Train data
# x_train = df_train['y_true']
# y_train = df_train['y_pred']
# u_train = df_train['sigma_pred']
# r2_train = r2_score(x_train, y_train)
# RMSE_train = sqrt(mean_squared_error(x_train, y_train))


# #Test data
# x_test = df_test['y_true']
# y_test = df_test['y_pred']
# u_test = df_test['sigma_pred']
def plot(x_test,y_test,save=True,xlab='True',ylab='Predicted',ax=None,marker_size=90,fontsize_label = 24,scale='linear',e_color='k',MARKER='s',density=False,unit='',density_params={'N_BINS':100,'MAP_NAME':'plasma_r','order':'random'},c='#0d78b3',lim=None,LABEL=None):
    '''
    x_test: true values
    y_test: predictions
    scale: 'linear', 'log'
    'order': 'random', 'dense_top', 'dense_bottom'
    '''
    # Font set
    mpl.rcParams['font.family'] = "arial narrow"
    fig_width = 8
    fig_height = 8
    fontsize_tick = 22
    fontsize_in_the_plot = 20
    factor = round(sqrt(marker_size / 90))
    
    if ax == None:
        plt.figure(figsize=(fig_width, fig_height)) 
        ax = plt.gca()
    
    #scale
    ax.set_yscale(scale)
    ax.set_xscale(scale)
    # Plot area
    if lim == None:
        lims = [0,1.2*max(x_test+y_test)]
    else:
        lims = [0,lim]
    
    r2_test = r2_score(x_test, y_test)
    RMSE_test = sqrt(mean_squared_error(x_test, y_test))
    

    # Train plot
    # plt.errorbar(x_train,y_train, yerr=u_train, fmt='o',  color='#9b000e', 
    #              markerfacecolor='#ff0018', capsize=2,elinewidth=1.1, linewidth=0.6, ms=9,
    #              label=' Train $R^2$= %.2f, RMSE= %.2f eV' %(r2_train, RMSE_train))

    # # Test plot
    u_test = [0 for i in x_test]
#     plt.errorbar(x_test,y_test, yerr=u_test, fmt='s',  color='k', 
#                  markerfacecolor='#0d78b3', capsize=2,elinewidth=1.1, linewidth=0.6, ms=9,
#                  label=' test $R^2$= %.2f, RMSE= %.2f eV' %(r2_test, RMSE_test))

    #c = '#0d78b3'
    if density:
        DENSITY_PARAMS = {'N_BINS':100,'MAP_NAME':'plasma_r','order':'random','count_thresh':None}
        for k in density_params.keys():
            DENSITY_PARAMS[k] = density_params[k]
        x_test, y_test, c, sm, count_data = ru.pltDensity(x_test,y_test,DENSITY_PARAMS['N_BINS'],DENSITY_PARAMS['MAP_NAME'],DENSITY_PARAMS['order'],DENSITY_PARAMS['count_thresh'])
#         if order != 'random':
#             if order == 'dense_top':
#                 REVERSE = False
#             else:
#                 REVERSE = True
                
#             inds_order = [y[0] for y in sorted(enumerate(c),key=lambda x: x[1],reverse=REVERSE)]
#             c = [c[i] for i in inds_order]
#             x_test = [x_test[i] for i in inds_order]
#             y_test = [y_test[i] for i in inds_order]
                
    if LABEL == None:
        LABEL = '$R^2$= %.2f, RMSE= %.2f %s' %(r2_test, RMSE_test,unit)
    
    ax.scatter(x_test,y_test, color=c,linewidths=1.1*factor,s=marker_size,marker=MARKER,edgecolor=e_color,
                label=LABEL)    
    # Parity line
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)

    # Legend
    if density:
        ax.annotate(LABEL, xy=(.95, 0.1), xycoords='axes fraction',
                size=fontsize_in_the_plot+2, ha='right', va='top',
                bbox=dict(boxstyle='round', fc='w'))
    else:
        ax.legend(loc='upper left',ncol=1, prop={'size': fontsize_in_the_plot+2},
                   handletextpad=1*factor,labelspacing=0.3*factor,columnspacing=1*factor,borderpad=1*factor, \
                  handlelength=1*factor)
    if density:
        #cbar = plt.colorbar(sm,ax=ax)
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins1 = inset_axes(ax,
                            width="100%",  # width = % of parent_bbox width
                            height="2%",  # height : 5%
                            bbox_to_anchor=(0., 0.04, 1, 1),
                            bbox_transform=ax.transAxes,
                            loc=9)
        im1 = ax.imshow([[1, 2], [2, 3]])
        print(max(count_data))
        cbar=plt.colorbar(sm, cax=axins1, orientation="horizontal",ticks=[0,max(count_data)-1])
        axins1.xaxis.set_ticks_position("top")
        #axins1.set_yticks([0,max(count_data)])
        cbar.set_label('Point Density', rotation=0,labelpad=8,size=fontsize_in_the_plot+2)
        cbar.ax.set_xticklabels(['Low','High'],rotation='horizontal')
        cbar.ax.tick_params(labelsize=fontsize_tick-2) 
#         if DENSITY_PARAMS['count_thresh'] != None:
#             cbar.ax.set_yticklabels(labels)

    ax.set_xlim(0,1.2*max(x_test+y_test))
    ax.set_ylim(0,1.2*max(x_test+y_test))

    #plt.xticks(np.linspace(0,12,6 , endpoint=False))
    #plt.yticks(np.linspace(0,12,6 , endpoint=False))

    ax.set_xlabel(xlab,size = fontsize_label)
    ax.set_ylabel(ylab,size = fontsize_label)
    
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(100))
    
    ax.tick_params(axis='both', which='major', direction='in', labelsize=fontsize_tick)

    plt.tight_layout()
    if save:
        #plt.savefig('parityPlot.eps', format='eps')#,dpi=600, bbox_inches = 'tight',    pad_inches = 0.1)
        plt.savefig('parityPlot.png', dpi=1200)
    #return plt.show()
    return ax