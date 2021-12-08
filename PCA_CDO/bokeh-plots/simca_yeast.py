# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 22:44:58 2018

@author: rolf
"""


import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cross_decomposition import PLSRegression, PLSCanonical, CCA
import matplotlib.pyplot as plt
from sklearn import datasets,preprocessing
import scipy
import h5py
from sklearn.preprocessing import StandardScaler
import pandas as pd

from bokeh.layouts import row
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, CustomJS, PolySelectTool, TapTool
from bokeh.io import show, output_file
from bokeh.models import FuncTickFormatter, FixedTicker
from bokeh.layouts import column, row
from bokeh.models import ColorBar, Circle

from bokeh.models import  Callback, ColumnDataSource, Rect, Select,CustomJS
from bokeh.plotting import figure, output_file, show,  gridplot

from sklearn.preprocessing import MinMaxScaler, RobustScaler


def plot_bokeh(y, X, average, upper, lower, DModX_upper, t2_upper, t2, t2_con, dx, dx_con, file_in):
    output_file(file_in)
    x = X['Y var (Time)'] # np.arange(len(average))
    colnames = np.array(t2_con.columns, dtype = np.str)
    cols = [colnames for x in range(len(average))]
    
    s1 = ColumnDataSource(dict(
        x = x,
        y = y,
        average = average,
        upper = upper,
        lower = lower,
        cols =cols,
        t2 = t2,
        dmodx_upper = DModX_upper,
        t2_upper = t2_upper,
        t2_con = [t2_con.iloc[i,].values for i in range(len(average))],
        dx = dx,
        dx_con = [dx_con.iloc[i,].values for i in range(len(average))]
        )
    )
    

    s2 = ColumnDataSource(data=dict(cols=[], t2_con=[], dx_con = []))
    
    X['y'] = X.iloc[:,1]
    s3 = ColumnDataSource(X)     
    
    #source_contrib = ColumnDataSource(df)
    
    hover = HoverTool(tooltips=[
        ("x", "@x")
        ])    


    #tap.callback = CustomJS(code='''alert("pressed");''')  
    
    p = figure(plot_width=600, plot_height=400, title = "Principal Component 1 for faulty batch",
                   x_axis_label = "Time [h]",
                   tools =[hover,'box_select','tap','reset'])
#                   tools =['tap','reset'])
                   
    # add a line renderer
    p.line(x='x', y='average', color = 'green', source = s1, line_width = 2)
    p.line(x='x', y='upper', color = 'red', source = s1, line_dash=[6, 3], line_width = 2)
    p.line(x='x', y='lower', color = 'red', source = s1, line_dash=[6, 3], line_width = 2)
    p.line(x='x', y='y', color = 'black', source = s1, line_width = 2)
    renderer = p.circle(x='x', y = 'y', color = 'black', fill_color="red", source = s1, size= 6)
    selected_circle = Circle( line_color = 'black', fill_color="red",  line_width = 2)
    nonselected_circle = Circle( fill_alpha=0.2, line_alpha=0.2, line_color = 'black', fill_color="red")
    renderer.selection_glyph = selected_circle
    renderer.nonselection_glyph = nonselected_circle
    
    

  
    
    
    #p.add_glyph(s1,selected_circle, selection_glyph=selected_circle,
    #        nonselection_glyph=nonselected_circle)
    
    
    # plot T2
    p_t2 = figure(plot_width=600, plot_height=400, title = "Hotellings T2",
                  x_axis_label = "Time [h]",
                  tools =[hover,'box_select','tap','reset'])  
    p_t2.line(x='x', y='t2', color = 'black', source = s1, line_width = 2)
    p_t2.circle(x='x', y = 't2', color = 'black', fill_color="red", source = s1, size=5)
    p_t2.line(x='x', y='t2_upper', color = 'red', source = s1, line_dash=[6, 3], line_width = 2) 
    
    # plot DModX
    p_dx = figure(plot_width=600, plot_height=400, title = "DModX",
                  x_axis_label = "Time [h]",
                  tools =[hover,'box_select','tap','reset'])  
    p_dx.line(x='x', y='dx', color = 'black', source = s1, line_width = 2)
    p_dx.circle(x='x', y = 'dx', color = 'black', fill_color="red", source = s1, size=5)
    p_dx.line(x='x', y='dmodx_upper', color = 'red', source = s1, line_dash=[6, 3], line_width = 2)    
    
    
    
    #plot contribution to PC1
    t2max = max(abs(np.max(t2_con.values)), abs(np.min(t2_con.values)))
    p2 = figure(x_range = colnames, title = "T2 Contributions", 
                plot_width=600, plot_height=400, y_range =(-t2max, t2max),
                tools =   ['hover','box_zoom','reset'])#, tools =['box_select','tap','reset'])
    p2.vbar(x='cols', top='t2_con', width = 0.9, source = s2 )
    p2.xaxis.major_label_orientation = 0.4*np.pi

    #plot contribution to DModX
    dxmax = max(abs(np.max(dx_con.values)), abs(np.min(dx_con.values)))
    p_dxcon = figure(x_range = colnames, title = "DModX Contributions",
                    y_range =(-dxmax, dxmax), plot_width=600, plot_height=400, 
                    tools =   ['hover','box_zoom','reset'])#, tools =['box_select','tap','reset'])
    p_dxcon.vbar(x='cols', top='dx_con', width = 0.9, source = s2 )
    p_dxcon.xaxis.major_label_orientation = 0.4*np.pi


# Add a hover tool, that selects the circle

    s1.callback = CustomJS(args=dict(s1=s1,s2=s2), code="""
            // only select one point
            var data = s1.data;
            var selected = s1.selected['1d']['indices'];
            if (selected.length > 0){
                select_inds = [selected[0]];
                s1.selected['1d']['indices'] = select_inds 
            } 
            
            
            s1.change.emit();
            
            // update the bar
            var inds = cb_obj.selected['1d'].indices;
            var d1 = cb_obj.data;
            var d2 = s2.data;
            
            //d2['cols'] = []
            d2['t2_con'] = 0*d2['t2_con'] 
            d2['dx_con'] = 0*d2['dx_con']
                
            if (selected.length > 0){
                d2['cols'] = []
                d2['t2_con'] = []
                d2['dx_con'] = []
            
                i=0
                for (j = 0; j < d1['cols'][inds[i]].length; j++) {
                    d2['cols'].push(d1['cols'][inds[i]][j])
                    d2['t2_con'].push(d1['t2_con'][inds[i]][j])
                    d2['dx_con'].push(d1['dx_con'][inds[i]][j])
                    }
                }
            s2.change.emit();
        """)
    
    
    
    
#    p_series = figure(plot_width=800, plot_height=400, title = "Selected Data Series",
#                   tools =[hover,'box_select','tap','reset'])
#    p_series.line(x='Time [h]', y='pH [–]', color = 'black', source = s3, line_width = 2)  
#    callback_drop = CustomJS(args={'s1': s1}, code="""
#        var f = cb_obj.get('value')   
#        """)        
#    dropdown = Select(title="Select Series", value=colnames[0], options= colnames,  callback = callback_drop)
#    #Display data
#    #filters = VBox(dropdown)


    # finally a selectbox and a plot
#    s3 = ColumnDataSource(X_plot)  
# 
    p_series = figure(plot_width=600, plot_height=330, title = colnames[1],
                   x_axis_label = "Time [h]",
                   tools =['box_select','tap','reset'])
    p_series.line(x='Y var (Time)', y='y', color = 'black', source = s3, line_width = 2)  
    p_series.circle(x='Y var (Time)', y='y', color = 'black', fill_color="red", source = s3, size=5)
#output_file('/home/rolf/bla.html')
#
#
    callback = CustomJS(args=dict(s3 = s3, plot = p_series), code="""
        // only select one point
        var data = s3.data;
        var cols = cb_obj.value;
        //data['y'] = [];
        data['y'] = data[cols];
        plot.title.text = cols;
        plot.change.emit();
        s3.change.emit();
        """)
#        
#        
##select.on_change('value', update_plot)
    colnames = list(np.array(X.columns[1:X.shape[1]-1], dtype = np.str))
    select = Select(value=colnames[1], title="Select Parameter", options=colnames)
#cols = colnames[1]
#
#
#
    select.js_on_change('value', callback)
#
#show(row(p_series, select))
    #show(p_series)
    l = column(row(p, column(select, p_series)), row(p_t2, p2), row(p_dx, p_dxcon) )

    show(l)
    
    
    
    
# read in the good batch data
xls_file = '/home/rolf/simca/BakersYeast.xls'
yeast_data = pd.read_excel(xls_file)
qual_data = pd.read_excel(xls_file,2)

bad = qual_data['QP2'] < 75
bad_list = list(qual_data[bad]['BatchID'])
bad_list.extend(['Ba', 'Aa', 'Ga', 'Oa', 'Pa', 'Ua'])#, 'lb'])



good_list = list(set(yeast_data['BatchID']) - set(bad_list))

good_list = ['Ba', 'Ca', 'Ia', 'Ma', 'Na', 'Qa', 'Ra', 'Ta', 'Va', 'Xa', 'Za', 'ab', 'bb', 'cb', 'db', 'eb', 'fb', 'gb', 'hb', 'ib']
#rearrange data for me
#for batch_id in np.unique(yeast_data.iloc[:,1]):
#    print(np.sum(yeast_data.iloc[:,1] == batch_id))
n_time = 83
n_batches = len(good_list)
Xdata = np.zeros([n_time, 8, n_batches])
X_rearranged = np.zeros([n_time*n_batches, 8])
k = 0
for batch_id in good_list:
    #print(np.sum(yeast_data.iloc[:,1] == batch_id))
    msk = (yeast_data.iloc[:,1] == batch_id)
    Xdata[:,:,k] = np.array(pd.DataFrame(yeast_data[msk]).iloc[:,2:])
    X_rearranged[k*n_time:((k+1)*n_time),:] = Xdata[:,:,k].copy()
    k+=1

#X_rearranged = np.array(yeast_data.iloc[:,2:])

#X_rearranged = Xdata.reshape(1743,-1, order = 'K')#-X_rearranged
#plt.plot(X_rearranged[:,0])

# read in bad batch data
n_batches = len(bad_list)
Xdata_bad = np.zeros([n_time, 8, n_batches])
k = 0
for batch_id in bad_list:
    #print(np.sum(yeast_data.iloc[:,1] == batch_id))
    msk = (yeast_data.iloc[:,1] == batch_id)
    Xdata_bad[:,:,k] = np.array(pd.DataFrame(yeast_data[msk]).iloc[:,2:])
    k+=1



    
column_names =np.array(yeast_data.columns, dtype = np.str)


def get_t2(X, pls_or_pca, scaler, X_org, variant = 0):

# http://wiki.eigenvector.com/index.php?title=T-Squared_Q_residuals_and_Contributions
# the different variants are from this paper:
# Total PLS Based Contribution Plots for Fault Diagnosis
# The contributions, calculated with the variant proposed in the paper (variant 4) 
# seems to give totally unrelated results. if you have time, pls check my implementation

    
    if ('pls' in str(type(pls_or_pca)) ) or ('cca' in str(type(pls_or_pca)) ) :
        is_pls = True
        pls = pls_or_pca
    if 'pca' in str(type(pls_or_pca)):
        is_pls = False
        pca = pls_or_pca
        

    # Dimensions and scaling
    n, I = X.shape
    X_scaled = scaler.transform(X.copy())
    X_scores = pls_or_pca.transform(X_scaled.copy())

        
    # intialize contrbutions
    t_con = np.zeros([n, I])

    # define some matrices
    if (is_pls):
        T = pls.x_scores_.copy()
        n_train, A = T.shape
        W = pls.x_weights_.copy()
        P = pls.x_loadings_.copy()
        lambda_ = 1/(n_train-1) * np.dot(T.T, T)
        eigenvalues = np.diagonal(lambda_.copy()).T
        
    if (not is_pls):
        T = scores = X_scores
        P = loadings = pca.components_ 
        eigenvalues = pca.explained_variance_   

        # pca contributions...this is clearly defined.
        for i in range(t_con.shape[0]):
            t_con[i,:] = np.dot(T[i,:], np.dot(np.diag(1.0/np.sqrt(eigenvalues)), P))
            
            
    variant =0
#    D = np.dot(P/eigenvalues,P.T)
#    S = np.dot(X_org.T,X_org)/(n-1)
#    DSD = np.dot(np.dot(D,S),D)
#    D2 = scipy.linalg.sqrtm(D)
#    a = np.dot(X_scaled, D)
#    print(S.shape)
#    for i in range(t_con.shape[0]):
#        for j in range(t_con.shape[1]):
#            #t_con[i,j] = X_scaled[i,j]**2/S[j,j]#np.dot(X_scaled[i,:], D)[j]*X_scaled[i,j] #X_scaled[i,:]*np.diag(D)*X_scaled[i,:]
#            t_con[i,j] = a[i,j]*X_scaled[i,j]/(np.dot(D,S)[j,j])            
#            #t_con[i,j] = a[i,j]**2/(np.dot(D,S)[j,j]) 
#            #t_con[i,j] = a[i,j]**2/DSD[j,j]
            
    if (is_pls):
        # see formula (12)
        if (variant == 0):
            for i in range(I):
                for a in range(1):
                    t_con[:,i] += X_scores[:,a]/eigenvalues[a]*P[i,a]*X_scaled[:,i]
        # see formula (13)
        if (variant == 1):            
            PTP = np.linalg.inv(np.dot(P.T,P))
            t_new = np.dot(PTP, np.dot(P.T, X_scaled.T)).T
            
            for i in range(I):
                t_con[:,i] = np.dot(np.dot( (t_new/eigenvalues) , PTP),  P[i,:])*X_scaled[:,i]
        # see formula (15)
        if (variant == 2):
            for i in range(I):
                for j in range(n):
                    t_con[j,i] = np.linalg.norm(X_scaled[j,i]* P[i,:]/np.sqrt(eigenvalues),2)**2
        # see formula (16)        
        if (variant == 3):  
            R = np.dot(W, np.linalg.inv(np.dot(P.T, W))) # can inv be a problem?
            for i in range(I):
                for j in range(n):
                    t_con[j,i] = (np.linalg.norm(R[i,:] /np.sqrt(eigenvalues)*X_scaled[j,i],2))**2
        # formula (17)
        if (variant == 4):
            R = np.dot(W, np.linalg.inv(np.dot(P.T, W))) # can inv be a problem?
            RTR = np.dot(R/eigenvalues, R.T)
            # test for almost zero
            zero_row = (np.sum(abs(RTR), axis = 1) < 1e-10)*1
            Gamma = scipy.linalg.sqrtm(RTR + np.diag(zero_row))
            #t_con = (np.dot(X_scaled, Gamma).real)**2
            for i in range(I):
                for j in range(n):
                    t_con[j,i] = (np.dot(X_scaled[j,:], Gamma[i,:]).real)**2
        if (variant == 5):
            D = np.dot(P/eigenvalues,P.T)
            D2 = scipy.linalg.sqrtm(D)
            t_con = np.dot(X_scaled, D2)
        if (variant == 6):
            D = np.dot(P/eigenvalues,P.T)
            D2 = scipy.linalg.sqrtm(D)
            a = np.dot(X_scaled, D2)
            for i in range(t_con.shape[0]):
                t_con[i,:] = a[i,:]*X_scaled[i,:]
        if (variant == 7):
            D = np.dot(P/eigenvalues,P.T)
            for i in range(t_con.shape[0]):
                t_con[i,:] = X_scaled[i,:]*np.diag(D)*X_scaled[i,:]
        if (variant == 8):
                D = np.dot(P/eigenvalues,P.T)
                for i in range(t_con.shape[0]):
                    for j in range(t_con.shape[1]):
                        t_con[i,j] = D[j,j]*X_scaled[i,j]**2
            
            
                    
    t2 = np.zeros(n)
    for j in range(n):
        #t2[j] = np.dot(np.dot(X_scores[j,:], np.diag(1.0/eigenvalues)),X_scores[j,:])
        t2[j] = np.sum(X_scores[j,:]/eigenvalues*X_scores[j,:])
#        t_con[i,:] = np.dot(scores[i,:], np.dot(np.diag(1.0/np.sqrt(eigenvalues)), loadings))
#       #j=3
#       #t_con[i,:] = scores[i,j]/np.sqrt(eigenvalues[j])*loadings[j,:].T
    #t22  = np.sum(t_con, axis = 1)#*t_con, axis=1)  
    

    #t2 = np.dot(pca.x_loadings_,np.diag(1.0/np.sqrt(eigenvalues) )

    return t2, t_con.real
    
def get_DModX(X, pls_or_pca, scaler): 
    
    # Dimensions and scaling
    n, I = X.shape
    X_scaled = scaler.transform(X.copy())
    X_scores = pls_or_pca.transform(X_scaled.copy())
    n_train, A = pls.x_scores_.shape #X_scores.shape
        
    # intialize contrbutions
    t_con = np.zeros([n, I])


    if 'pls' in str(type(pls_or_pca)):
        is_pls = True
    if 'pca' in str(type(pls_or_pca)):
        is_pls = False        
        
    # define some matrices
    if (is_pls):
        T = pls_or_pca.x_scores_
        W = pls_or_pca.x_weights_
        P = pls_or_pca.x_loadings_
        R = np.dot(W, np.linalg.inv(np.dot(P.T, W)))
        lambda_ = 1/(n_train-1) * np.dot(T.T, T)
        eigenvalues = np.diagonal(lambda_).T
        
    if (not is_pls):
        T = X_scores
        P = pls_or_pca.components_ # loadings
        eigenvalues = pls_or_pca.explained_variance_   
        
    
    # Dimensions and scaling
    #n_train, A = pls.x_scores_.shape
    #n, I = X.shape
    #X_scaled = scaler.transform(X)
    #X_scores = pls.transform(X_scaled)
    
    if is_pls:
        DModX_con = X_scaled - np.dot(np.dot(P,R.T), X_scaled.T).T
    if (not is_pls):
        DModX_con = X_scaled - np.dot(T,P)
    
    DModX = np.linalg.norm(DModX_con, axis = 1)#/np.sqrt((I-A)/A)
        
        
    return DModX, DModX_con
 
 
    
    
# select number of compone 
ncomp = 3

# selct columns
indices = [1,2,3,4,5,6,7]

#batch_no = 10
#xbad_u = Xdata_bad[:,indices, batch_no]
msk = (yeast_data.iloc[:,1] == 'Ja')
xbad_u = np.array(pd.DataFrame(yeast_data[msk]).iloc[:,3:])
#msk = (yeast_data.iloc[:,1] == 'db')
#xbad_u = np.array(pd.DataFrame(yeast_data[msk]).iloc[:,3:])
#msk = (yeast_data.iloc[:,1] == good_list[0])
#xbad_u = np.array(pd.DataFrame(yeast_data[msk]).iloc[:,3:].copy())



# define the scaler for PLS and PCA

scaler = StandardScaler(with_mean=True, with_std=True, copy=True)

#scaler = MinMaxScaler()
#scipy.stats.boxcox(X_rearranged[:,2])
#for i in indices:
#    mini = np.min(np.hstack([xbad_u[:,i-1], X_rearranged[:,i]])) 
#    lmbda = scipy.stats.boxcox_normmax(X_rearranged[:,i] - mini + 1e-8)
#    X_rearranged[:,i] = scipy.stats.boxcox(X_rearranged[:,i] - mini + 1e-8, lmbda)
#    xbad_u[:,i-1] = scipy.stats.boxcox(xbad_u[:,i-1] - mini + 1e-8, lmbda)

xgood = X_rearranged[:,indices]
scaler.fit(X_rearranged[:,indices])
X = scaler.transform(X_rearranged[:,indices])
y_pls = X_rearranged[:,0]



#scalery = StandardScaler(copy=True, with_mean=True, with_std=True)
#y_pls = scalery.fit_transform(X_rearranged[:,0].reshape(-1,1))


# use PLS, data is already scaled
pls = PLSRegression(n_components=ncomp, scale = False, copy=True)
#pls = CCA(n_components=ncomp, scale = False)

pls.fit(X = X, Y = y_pls)
X_pls_pca = pls.x_scores_#pls.transform(X, copy=True)
y_pred = pls.predict(X, copy=True)



xbad_s = scaler.transform(xbad_u)
xbad = pls.transform(xbad_s)


# calc std
std_array = np.zeros([Xdata.shape[0], ncomp])
avg_array = np.zeros([Xdata.shape[0], ncomp])
for n in range(ncomp):
    std_array[:,n] = X_pls_pca[:,n].reshape(Xdata.shape[0],-1, order='F').std(axis=1)
    avg_array[:,n] = X_pls_pca[:,n].reshape(Xdata.shape[0],-1, order='F').mean(axis=1)


# calc for DModX
DModX, DModX_con = get_DModX(xgood.copy(), pls, scaler)
DModXstd = DModX.reshape(Xdata.shape[0],-1, order='F').std(axis=1)
DModXmean = DModX.reshape(Xdata.shape[0],-1, order='F').mean(axis=1)
DModX_upper = DModXmean + 3*DModXstd

# calc for t2
t2_all, t2_con_all = get_t2(xgood, pls, scaler, X, variant = 1)
t2std = t2_all.reshape(Xdata.shape[0],-1, order='F').std(axis=1)
t2mean = t2_all.reshape(Xdata.shape[0],-1, order='F').mean(axis=1)
t2_upper = t2mean + 3*t2std






# TESTS   
#t2, t2_con = get_t2_pls(xbad_u, pls, scaler, variant = 4)
#plt.plot(t22)
#plt.plot(t2,'r.')
#DModX, DModX_con = get_DModX_pls(xbad_u, pls, scaler)

t2, t2_con = get_t2(xbad_u.copy(), pls, scaler, X, variant = 3)
DModX, DModX_con = get_DModX(xbad_u.copy(), pls, scaler)

t2_con = pd.DataFrame(t2_con)
t2_con.columns = np.array(column_names)[2:][indices]
DModX_con = pd.DataFrame(DModX_con)

# define what to plot
pr_comp = 0
avg = avg_array[:,pr_comp]
upper = avg_array[:,pr_comp] + 3.*std_array[:,pr_comp]
lower = avg_array[:,pr_comp] - 3.*std_array[:,pr_comp]


plt.plot(upper); plt.plot(lower)


file_pls = "/home/rolf/pls.html"
X_plot = pd.DataFrame(yeast_data[msk]).iloc[:,2:].copy()
X_plot.columns = yeast_data.columns[2:]
plot_bokeh(xbad[:,pr_comp], X_plot, avg, upper, lower, DModX_upper, t2_upper, t2, t2_con, DModX, DModX_con, file_pls)
#plot_bokeh(y, average, upper, lower, t2, t2_con, dx, dx_con, file_in):
 
#
#
################ PCA ##########################
#pca = PCA(n_components=ncomp, svd_solver = "full")
#pca.fit(X)
#X_pls_pca = pca.transform(X)
#
## calc std
#std_array = np.zeros([Xdata.shape[0], ncomp])
#avg_array = np.zeros([Xdata.shape[0], ncomp])
#for n in range(ncomp):
#    std_array[:,n] = X_pls_pca[:,n].reshape(Xdata.shape[0],-1, order='F').std(axis=1)
#    avg_array[:,n] = X_pls_pca[:,n].reshape(Xdata.shape[0],-1, order='F').mean(axis=1)
#
#t2, t2_con = get_t2(xbad_u, pca, scaler)
#DModX, DModX_con = get_DModX(xbad_u, pca, scaler)
#
#t2_con = pd.DataFrame(t2_con)
#t2_con.columns = np.array(column_names)[indices]
#DModX_con = pd.DataFrame(DModX_con)
#
## define what to plot
#pr_comp = 1
#xbad = pca.transform(scaler.transform(Xdata_bad[:,indices,batch_no]))
#avg = avg_array[:,pr_comp]
#upper = avg_array[:,pr_comp] + 3*std_array[:,pr_comp]
#lower = avg_array[:,pr_comp] - 3*std_array[:,pr_comp]
#
#
#file_pca = "/home/rolf/pca.html"
#
#X_plot = Xdata_bad[:,:,batch_no]
#X_plot = pd.DataFrame(X_plot)
#X_plot.columns = np.array(column_names[2:])
#
##plot_bokeh(xbad[:,pr_comp], avg, upper, lower, t2_con, file_pca)
#plot_bokeh(xbad[:,pr_comp], X_plot, avg, upper, lower, t2, t2_con, DModX, DModX_con, file_pca)
#
# 
#
#s3 = ColumnDataSource(X)  
# 
#p_series = figure(plot_width=800, plot_height=400, title = "Selected Data Series",
#                   tools =[hover,'box_select','tap','reset'])
#p_series.line(x='Time [h]', y='pH [–]', color = 'black', source = s3, line_width = 2)  
#callback_drop = CustomJS(args={'s1': s1}, code="""
#        var f = cb_obj.get('value')   
#        """)        
#dropdown = Select(title="Select Series", value=colnames[0], options= colnames,  callback = callback_drop)
#    #Display data
#    #filters = VBox(dropdown)
# 
# 
# 
# 
# 
#colnames = list(np.array(X_plot.columns, dtype = np.str))
#select = Select(value=colnames[10], title="Select Series", options=colnames)
#cols = colnames[10]
#
#
#X_new = X_plot[[colnames[0],colnames[10]]]
#X_new.columns = ['Time [h]', 'y']
##df = pd.read_csv(join(dirname(__file__), 'data/2015_weather.csv'))
##source = get_dataset(df, cities[city]['airport'], distribution)
##plot = make_plot(source, "Weather data for " + cities[city]['title'])
#
#s3 = ColumnDataSource(X_new)  
# 
#p_series = figure(plot_width=800, plot_height=400, title = "Selected Data Series",
#                   tools =['box_select','tap','reset'])
#p_series.line(x='Time [h]', y='y', color = 'black', source = s3, line_width = 2)  
#
#
#
#def update_plot(attrname, old, new):
#    cols = select.value
#    p_series.title.text = cols
#
#    idx = np.where(X_plot.columns == cols)[0][0]
#    X_new = X_plot[[colnames[0],colnames[idx]]]
#    X_new.columns = ['Time [h]', 'y']
#
#
#    src = ColumnDataSource(X_new)
#    s3.data.update(src.data)
#
#
#
#
#select.on_change('value', update_plot)
#
#
#
#show(row(p_series, select))
# 
# 
 
 
 
 
 
 