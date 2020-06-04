import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt


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
def plot(x_test,y_test,save=True,xlab='True',ylab='Predicted'):
    '''
    x_test: true values
    y_test: predictions
    '''
    # Font set
    mpl.rcParams['font.family'] = "arial narrow"
    fig_width = 8
    fig_height = 8
    fontsize_label = 24
    fontsize_tick = 22
    fontsize_in_the_plot = 20


    # Plot area
    lims = [0,1.1*max(x_test)]
    
    r2_test = r2_score(x_test, y_test)
    RMSE_test = sqrt(mean_squared_error(x_test, y_test))
    plt.figure(figsize=(fig_width, fig_height)) 

    # Train plot
    # plt.errorbar(x_train,y_train, yerr=u_train, fmt='o',  color='#9b000e', 
    #              markerfacecolor='#ff0018', capsize=2,elinewidth=1.1, linewidth=0.6, ms=9,
    #              label=' Train $R^2$= %.2f, RMSE= %.2f eV' %(r2_train, RMSE_train))

    # # Test plot
    u_test = [0 for i in x_test]
#     plt.errorbar(x_test,y_test, yerr=u_test, fmt='s',  color='k', 
#                  markerfacecolor='#0d78b3', capsize=2,elinewidth=1.1, linewidth=0.6, ms=9,
#                  label=' test $R^2$= %.2f, RMSE= %.2f eV' %(r2_test, RMSE_test))

    plt.scatter(x_test,y_test, color='#0d78b3',linewidths=1.1,s=90,marker='s',edgecolor='k',
                label=' test $R^2$= %.2f, RMSE= %.2f eV' %(r2_test, RMSE_test))    
    # Parity line
    plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0)

    # Legend
    plt.legend(loc='upper left',ncol=1, prop={'size': fontsize_in_the_plot+2},
               handletextpad=1,labelspacing=0.3,columnspacing=1,borderpad=1,handlelength=1)


    plt.xlim(lims)
    plt.ylim(lims)

    #plt.xticks(np.linspace(0,12,6 , endpoint=False))
    #plt.yticks(np.linspace(0,12,6 , endpoint=False))



    plt.xlabel(xlab,size = fontsize_label)
    plt.ylabel(ylab,size = fontsize_label)

    plt.tick_params(axis='both', which='major', direction='in', labelsize=fontsize_tick)

    plt.tight_layout()
    if save:
        #plt.savefig('parityPlot.eps', format='eps')#,dpi=600, bbox_inches = 'tight',    pad_inches = 0.1)
        plt.savefig('parityPlot.png', dpi=1200)
    #return plt.show()
    return plt.gca()