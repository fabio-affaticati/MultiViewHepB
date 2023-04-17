
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from statsmodels.tools.sm_exceptions import ConvergenceWarning

RESULTS_DIR = './results/'


def ab_titers_inspection(ab_data, responder_classes, colors):

    ab = ab_data.copy()
    ab['Time_Point'] = [str(x) for x in ab['Time_Point']]
    ab['Log_AB_titer'] = np.log(ab['Antibody_titre'])
    pivoted = ab.pivot(index='Vaccinee', columns='Time_Point', values='Log_AB_titer')
    pivoted = pd.merge(pivoted, responder_classes, on='Vaccinee')

    pivoted_parallel = pivoted.copy()
    pivoted_parallel.rename(columns = {'0':'Day 0', '60':'Day 60', '180':'Day 180', '365':'Day 365'},inplace=True)
    
    fig, ax = plt.subplots(ncols=1, figsize=(8, 6))
    fig = pd.plotting.parallel_coordinates(pivoted_parallel[['Day 0','Day 60','Day 180','Day 365','Class']], class_column = 'Class', color=colors, ax=ax, lw=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.legend(bbox_to_anchor=(.7, 1))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.savefig(RESULTS_DIR + 'parallel_coordinates.png', dpi=600)
    plt.show()


    return pivoted


def plot_projection(title, data, colors, responder_classes = None):

    if title.endswith('pca'):

        pca_data = data.drop(['Vaccinee','Class'], axis=1)
        pca_data.reset_index(drop = True, inplace = True)

        scaler = StandardScaler()
        transf = scaler.fit_transform(pca_data)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(transf)

        df = pd.DataFrame({'pca-one' : pca_result[:,0], 'pca-two' : pca_result[:,1],
                        'label' : data['Class'], 'Vaccinee' : data['Vaccinee']})

        fig = px.scatter(df, x='pca-one', y='pca-two', color='label', hover_data=['Vaccinee'], symbol='label',
                        color_discrete_sequence = colors, title="",
                        labels={"pca-one": "PC1 (var = %.2f)" % pca.explained_variance_ratio_[0],
                                "pca-two": "PC2 (var = %.2f)" % pca.explained_variance_ratio_[1]},
                        category_orders={"label": ["Early-converter", "Late-converter", "Non-converter"]})
        
    elif title.endswith('mcca'):

        df = pd.DataFrame({'Component 1' : data[:,0], 'Component 2' : data[:,1],
                        'label' : responder_classes['Class'], 'Vaccinee' : responder_classes['Vaccinee']})
        
        fig = px.scatter(df, x='Component 1', y='Component 2', color='label', hover_data=['Vaccinee'], symbol='label',
                color_discrete_sequence = colors, title="",
                category_orders={"label": ["Early-converter", "Late-converter", "Non-converter"]})
                    


    fig.update_layout(font=dict(
        size=20), height=600, width=800,  legend=dict(y=0.8, x=0.9),
        legend_title="")
    fig.update_traces(marker=dict(size=15,))
    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    fig.write_image(RESULTS_DIR + title + ".png", scale = 4)
    fig.show()


def age_dist_plot(pivoted, colors):
    
    sns.set(font_scale = 1.5)

    sns.displot(pivoted, x="Age", hue="Class", kind="kde", fill=True, palette={clas: color for clas, color in zip(pivoted['Class'].unique(), colors)})

    plt.savefig(RESULTS_DIR + 'agedist.png', dpi=600)
    

def ab_titers_timedep(ab_data, colors):
    
    linear = ab_data.copy()
    linear['Antibody_titre'] = np.log(linear['Antibody_titre'])
    sns.set(font_scale = 1.5, rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

    warnings.simplefilter('ignore', ConvergenceWarning)
    ax = sns.lmplot(x="Age", y="Antibody_titre", hue="Time_Point",
                    col='Time_Point', robust=True,
                    data=linear, markers=['^', 'x', '+', '*'])

    for g, time in zip(ax.axes.flat, ax.col_names):
        r, pvalue = stats.pearsonr(linear[linear['Time_Point'] == time]['Age'], linear[linear['Time_Point'] == time]['Antibody_titre'])
        g.collections[0].set_label(f'R = {r:.2f} \npvalue = {pvalue:.3f}')
        g.legend()
        
    ax.fig.set_figheight(6)
    ax.fig.set_figwidth(12)

    plt.grid(axis='x')
    plt.savefig(RESULTS_DIR + 'AB_trends.png', dpi=600)


def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"


def cellcounts_dist_plot(cellcounts, labels, colors):

    cellcounts_unimodal = cellcounts.merge(labels, on='Vaccinee')

    ### Hand picked annotation positions
    pos_y = [11.6, 6.05, 17.5, 53, 480, 55, 9.6, 88.5]
    pos_yp = [11.9, 6.135, 17.86, 54, 492, 56.6, 9.88, 90]


    pvalues = [stats.mannwhitneyu(cellcounts_unimodal[col][cellcounts_unimodal['Class'] == 'Early-converter' ],
                cellcounts_unimodal[col][ cellcounts_unimodal['Class'] == 'Late-converter' ])[1]
                for col in cellcounts_unimodal.columns if not col in ['Vaccinee','Class']]
    print(pvalues)

    fig = make_subplots(rows=1, cols=8, shared_xaxes=True, horizontal_spacing = 0.05)

    for i, col in enumerate(cellcounts_unimodal.columns):
        if not col in ['Vaccinee','Class']:
            fig.add_trace(go.Violin(
                                    y=cellcounts_unimodal[col][cellcounts_unimodal['Class'] == 'Early-converter' ],
                                    legendgroup='Early-converter', scalegroup=col, name=col,
                                    side='negative',
                                    pointpos=-1.1,
                                    line_color=colors[0]), row= 1, col = i,
                        )
            fig.add_trace(go.Violin(
                                    y=cellcounts_unimodal[col][ cellcounts_unimodal['Class'] == 'Late-converter' ],
                                    legendgroup='Late-converter',scalegroup=col, name=col,
                                    side='positive',
                                    pointpos=1.1,
                                    line_color=colors[1],
                                    ), row= 1, col = i,
                        )
                        
            fig.add_shape(type="line",
                x0=-.5, y0=pos_y[i-1],
                x1=.5, y1=pos_y[i-1],
                line=dict(
                    color="black", width=1,
                ), row= 1, col = i,
            )

    annotations = [dict(x=0, y=pos_yp[i-1], yref='y'+str(i),xref='x'+str(i), text=f'p-value = {convert_pvalue_to_asterisks(np.round(pvalues[i-1],3))}',showarrow=False, ax=0, ay=0)
                    for i,col in enumerate(cellcounts_unimodal.columns) if not col in ['Vaccinee','Class']]

    fig.update_traces(meanline_visible=True,
                points='all', 
                jitter=0.05, 
                scalemode='count')

    fig.update_layout(font=dict(size=15),
                violingap=0, violingroupgap=0, violinmode='overlay',
                height=600, width=1400, showlegend=False,
                annotations = annotations)

    fig.write_image(RESULTS_DIR + "cellcounts_dist.png", scale = 4)
    fig.show()


def metadata_dist_plot(meta_hep, labels, colors):
    
    meta_hep_unimodal = pd.merge(labels, meta_hep, on ='Vaccinee', how = 'outer')    

    ### Hand picked annotation positions
    pos_y = [62.1, 37.9, 179.3, 129]
    pos_yp = [63.6, 37.99, 182, 131.5]

    pvalues = [stats.mannwhitneyu(meta_hep_unimodal[col][meta_hep_unimodal['Class'] == 'Early-converter' ],
                meta_hep_unimodal[col][ meta_hep_unimodal['Class'] == 'Late-converter' ])[1]
                for col in meta_hep_unimodal.columns if not col in ['Vaccinee','Class', 'Gender']]

    fig = make_subplots(rows=1, cols=4,shared_xaxes=True,)

    i = 1
    for col in meta_hep_unimodal.columns:
        if not col in ['Vaccinee','Class', 'Gender']:
            fig.add_trace(go.Violin(
                                    y=meta_hep_unimodal[col][ meta_hep_unimodal['Class'] == 'Early-converter' ],
                                    legendgroup='Early-converter', scalegroup=col, name=col,
                                    side='negative',
                                    pointpos=-1.1,
                                    line_color=colors[0]), row = 1, col = i
                            )
            fig.add_trace(go.Violin(
                                    y=meta_hep_unimodal[col][ meta_hep_unimodal['Class'] == 'Late-converter' ],
                                    legendgroup='Late-converter', scalegroup=col, name=col,
                                    side='positive',
                                    pointpos=1.1,
                                    line_color=colors[1],
                            ), row= 1, col = i,
                        
                        )
            fig.add_shape(type="line",x0=1.5, y0=pos_y[i-1],
                            x1=2.5, y1=pos_y[i-1],
                            line=dict(
                                color="black",
                                width=1,
                            ), row= 1, col = i,
                        ) 
            i+=1


    annotations = []    
    for i, col in enumerate(meta_hep_unimodal.columns[-4:]):
        
        annotations.append(
            dict(
                x=2, y=pos_yp[i],
                yref='y'+str(i+1),
                xref='x'+str(i+1),
                text=f'p-value = {np.round(pvalues[i],3)}',
                showarrow=False,
                ax=0,
                ay=0
            ))

            
    fig.update_traces(meanline_visible=True,
                points='all', 
                jitter=0.05, 
                scalemode='count')

    fig.update_layout(font=dict(size=15),
                violingap=0, violingroupgap=0, violinmode='overlay',
                height=600, width=1000, showlegend=False,
                annotations = annotations)


    fig.update_xaxes(categoryorder='array', categoryarray= ["Early-converter","Late-converter"])
    fig.write_image(RESULTS_DIR + "metadata_dist.png", height=600, width=1000, scale = 4) 
    fig.show()


def TCRseq_dist_plot(TCRseq, labels, colors):

    TCRseq_unimodal = pd.merge(labels, TCRseq, on ='Vaccinee', how = 'outer')
    fig = make_subplots(rows=1, cols=4,shared_xaxes=True,)

    ### Hand picked annotation positions
    pos_y = [107000, 0.076, 0.056, 1.5]
    pos_yp = [110250, 0.0786, 0.0578, 1.545]

    pvalues = [stats.mannwhitneyu(TCRseq_unimodal[col][TCRseq_unimodal['Class'] == 'Early-converter' ],
                TCRseq_unimodal[col][ TCRseq_unimodal['Class'] == 'Late-converter' ])[1]
                for col in TCRseq_unimodal.columns if not col in ['Vaccinee','Class']]
    print(pvalues)

    i = 1
    for col in TCRseq_unimodal.columns:
        if not col in ['Vaccinee','Class']:
            fig.add_trace(go.Violin(
                                    y=TCRseq_unimodal[col][ TCRseq_unimodal['Class'] == 'Early-converter' ],
                                    legendgroup='Early-converter', scalegroup=col, name=col,
                                    side='negative',
                                    pointpos=-1.1,
                                    line_color=colors[0]), row = 1, col = i
                            )
            fig.add_trace(go.Violin(
                                    y=TCRseq_unimodal[col][ TCRseq_unimodal['Class'] == 'Late-converter' ],
                                    legendgroup='Late-converter', scalegroup=col, name=col,
                                    side='positive',
                                    pointpos=1.1,
                                    line_color=colors[1],
                            ), row= 1, col = i,
                        
                        )
            fig.add_shape(type="line", x0=-0.5, y0=pos_y[i-1],
                    x1=0.5, y1=pos_y[i-1],
                    line=dict(
                        color="black",
                        width=1,
                    ), row= 1, col = i,
                )
            i+=1


    annotations = []    
    for i, col in enumerate(TCRseq_unimodal.columns[2:]):
        
        annotations.append(
            dict(
                x=0, y=pos_yp[i],
                yref='y'+str(i+1),
                xref='x'+str(i+1),
                text=f'p-value = {np.round(pvalues[i],3)}',
                showarrow=False,
                ax=0,
                ay=0
            ))
            
    fig.update_traces(meanline_visible=True,
            points='all', 
            jitter=0.05, 
            scalemode='count')

    fig.update_layout(font=dict(size=15),
                violingap=0, violingroupgap=0, violinmode='overlay',
                height=600, width=1000, showlegend=False,
                annotations = annotations)

    fig.write_image(RESULTS_DIR + "tcr_dist.png", height=600, width=1000, scale = 4)
    fig.show()


def plot_cor_matrix(corr, mask=None):
    
    f, ax = plt.subplots(figsize=(20, 12))
    sns.heatmap(round(corr,3), ax=ax,
                mask=mask,
                annot=True, vmin=-1, vmax=1, center=0, linewidths=.5,
                cmap='coolwarm', linecolor='white', cbar_kws={'orientation': 'vertical'})
    plt.xticks(rotation=45)   
    plt.savefig(RESULTS_DIR + 'corrplot.png', dpi=600)
    plt.show()      


def latent_contributions_plot(Xs, proj, title):

    for i in range(2):

        contributions = []
        features = []

        if title == 'MCCA':
            for view,_ in enumerate(proj.loadings_):
                contributions.extend(proj.loadings_[view][:,i])
                features.extend(list(Xs[view].columns))

        elif title == 'GroupPCA':
            for view,_ in enumerate(proj.individual_components_):
                contributions.extend(proj.individual_components_[view].T[:,i])
                features.extend(list(Xs[view].columns))

        contributions = np.array(contributions)

        res = pd.DataFrame({'Contributions' : contributions, 'Feature' : features})
        res['Function'] = [row.rsplit('.')[-1]+'.m' if len(row.rsplit('.')) > 1 else row.rsplit('.')[-1] for row in res['Feature']]
        
        res = res[(np.abs(res['Contributions']) > 0.001) & (res['Function'] != 'TBD.m')]
        res.reset_index(drop = True, inplace=True)
        res.sort_values(by=['Function'], ascending =False, inplace=True)

        fig = px.bar(res, x="Contributions", y="Function", color = 'Feature', orientation='h', title=f"{title} stacked contributions for Component {i+1}")
        
        fig.update_layout(
            font=dict(family="Grandview Display", size=20, color="black"),
                showlegend=False,plot_bgcolor='rgb(255, 255, 255)', width=800, height=1500,
            )
        fig.update_xaxes(zerolinewidth=1, zerolinecolor='black')
        fig.update_yaxes(gridcolor='grey', gridwidth = .5)

        fig.write_image(RESULTS_DIR + f'/{title}_contributions_Dim{i+1}.png', scale=4)
        fig.show()


def circle_plot(Xs, red_data, classes, title):

    classi = ['goldenrod' if col == 'Early-converter' else '#2ca02c' for col in classes]


    fig = go.Figure()
    ccircle = []
    feature_names = []


    if title == 'MCCA':
        
        selection = pd.concat([Xs[-1].filter(regex='Inflammation'), Xs[-1].filter(regex='Neutrophils')], axis=1)
        select = [Xs[-1].columns.get_loc(c) for c in selection if c in Xs[-1]]

        for k,view in enumerate(Xs):
            
            opacity = 1

            if k == 3:
                opacity = 0.7
                for col in view.columns[select]:
                    corr1 = np.corrcoef(view[col],red_data[:,0])[0,1]
                    corr2 = np.corrcoef(view[col],red_data[:,1])[0,1]
                    if (stats.pearsonr(view[col],red_data[:,0])[1] < 0.05) or (stats.pearsonr(view[col],red_data[:,1])[1] < 0.05):
                        ccircle.append((corr1, corr2))
                        feature_names.append(col)
                
            else:
                for col in view.columns:
                    corr1 = np.corrcoef(view[col],red_data[:,0])[0,1]
                    corr2 = np.corrcoef(view[col],red_data[:,1])[0,1]
                    ccircle.append((corr1, corr2))
                    feature_names.append(col)
                    
                    
                    
            list_of_all_arrows = []            
            for i,(x0,y0,x1,y1) in enumerate(zip(ccircle, ccircle, np.zeros(len(ccircle),), np.zeros(len(ccircle),))):
                text = feature_names[i]
                textangle = -45
                if (y0[1]<0) and (x0[0]<0):
                    textangle = 135
                elif (y0[1]>0) and (x0[0]<0):
                    textangle = 45
                elif (y0[1]<0) and (x0[0]>0):    
                    textangle = 45
                    
                    
                if (text == 'Age') or (text == 'HepBTCRs') or (text == 'MON0') or (text == 'RBC0') or (text == 'HGB0') or (text == 'HCT0') or (text == 'GRA0') or (text == 'Temperature') or (text.endswith('Inflammation')) or (text.endswith('Neutrophils')):
                    col = k
                    if (k == 3) and (text.endswith('Neutrophils')):
                        col+=1
                    arrow = go.layout.Annotation(dict(
                                    ax=x0[0],
                                    ay=y0[1],
                                    xref="x", yref="y",
                                    text=text,
                                    align='center',
                                    showarrow=True,
                                    axref="x", ayref='y',
                                    x=x1,
                                    y=y1,
                                    opacity = opacity,
                                    font=dict(
                                    family="Helvetica",
                                            size=12,
                                            color=px.colors.qualitative.D3[-col]
                                            ),
                                    textangle = textangle,
                                    arrowhead=0,
                                    arrowwidth=1.5,
                                    arrowcolor=px.colors.qualitative.D3[-col]
                                    )
                                )
                    list_of_all_arrows.append(arrow)
            
            fig.update_layout(annotations=list_of_all_arrows)
            
            fig.add_trace(go.Scatter(
            x=red_data[:,0], y=red_data[:,1],
            mode='markers',
            showlegend = False,
            marker=dict(color=classi, size = list(Xs[2]['Age']//2)),
            text=list(Xs[2]['Age'])
                ))

    elif title == 'PCA':

        for i, col in enumerate(Xs.columns):
            corr1 = np.corrcoef(Xs[col],red_data[:,0])[0,1]*20
            corr2 = np.corrcoef(Xs[col],red_data[:,1])[0,1]*20
            ccircle.append((corr1, corr2))
            feature_names.append(col)
                    
        list_of_all_arrows = []     

        for i,(x0,y0,x1,y1) in enumerate(zip(ccircle, ccircle, np.zeros(len(ccircle),), np.zeros(len(ccircle),))):

            text = feature_names[i]
            textangle = -45
            if (y0[1]<0) and (x0[0]<0):
                textangle = -45
            elif (y0[1]>0) and (x0[0]<0):
                textangle = 45
            elif (y0[1]<0) and (x0[0]>0):    
                textangle = 45

            flag = True
            if (text == 'Age') or (text == 'Temperature'):
                col = 2
            elif (text == 'HepBTCRs'):
                col = 1 
            elif (text == 'MON0') or (text == 'RBC0') or (text == 'HGB0') or (text == 'HCT0') or (text == 'GRA0'):
                col = 0
            elif (text == 'M14.48.Inflammation') or (text == 'M14.50.Inflammation')  or (text == 'M15.109.Inflammation'):
                col = 3
            elif (text == 'M13.22.Neutrophils') or (text == 'M15.35.Neutrophils'):
                col = 4
            else:
                flag = False
            if flag:
                arrow = go.layout.Annotation(dict(
                                ax=x0[0],
                                ay=y0[1],
                                xref="x", yref="y",
                                text=text,
                                align='center',
                                showarrow=True,
                                axref="x", ayref='y',
                                x=x1,
                                y=y1,
                                opacity = .9,
                                font=dict(
                                family="Helvetica",
                                        size=12,
                                        color=px.colors.qualitative.D3[-col]
                                        ),
                                textangle = textangle,
                                arrowhead=0,
                                arrowwidth=1.5,
                                arrowcolor=px.colors.qualitative.D3[-col]
                            )
                            )
                list_of_all_arrows.append(arrow)


        fig.update_layout(annotations=list_of_all_arrows)

        fig.add_trace(go.Scatter(
            x=red_data[:,0], y=red_data[:,1],
            mode='markers',
            showlegend = False,
            marker=dict(color=classi, size = list(Xs['Age']//2)),
            text=list(Xs['Age'])
                ))
        

    fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="Early-converter",
                marker=dict(size=7, color="goldenrod"),
            ))
    fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="Late-converter",
                marker=dict(size=7, color="#2ca02c"),
            ))

    fig.update_xaxes(
        zerolinewidth=1, zerolinecolor='grey',
        constrain="domain",
    )
    fig.update_yaxes(
        zerolinewidth=1, zerolinecolor='grey',
        scaleanchor = "x",
        scaleratio = 1
    )
    fig.update_layout(
        yaxis_visible=True, yaxis_showticklabels=False,
        xaxis_visible=True, xaxis_showticklabels=False,
        legend_title_text='',
        width = 800,
        height = 800,
        xaxis_title="Component 1",
        yaxis_title="Component 2",
            plot_bgcolor='rgb(255, 255, 255)')
    
    fig.write_image(RESULTS_DIR + f'/{title}_circleplot.png', scale=4)
    fig.show()


def unimodal_featimp_plot(Xs, coefficients):

    _, ax = plt.subplots(4, 1, sharex=True, gridspec_kw={'height_ratios': [1.5,1,1,4.5]}, figsize=(10, 16) )
    sns.set_palette("deep")
    for i in range(4):
        plt.sca(ax[i])
        contributions = coefficients[i::4]
        contributions = pd.DataFrame([item for sublist in contributions for item in sublist])
        contributions.columns = Xs[i].columns
        contributions = contributions.reindex(contributions.median().sort_values(ascending = False).index, axis=1)
        if i == 3:
            # remove TBD modules
            contributions = contributions[contributions.columns.drop(list(contributions.filter(regex='.TBD')))]
            contributions.columns = [x.split('.')[-1] for x in list(contributions.columns)]
            contributions = contributions.groupby(contributions.columns, axis=1).sum()
            contributions = contributions.reindex(contributions.median().sort_values(ascending = False).index, axis=1)

        
        sns.boxplot(data=contributions, showfliers=False, orient="h")
        
        if i == 3: 
            s = 1.5
        else: 
            s = 3
        
        sns.stripplot(data=contributions, color="0.25", s=s, orient="h")

        plt.ylabel("Feature")
        plt.grid(axis='x')
        plt.grid(axis='y')
        plt.axvline(0, color='red')

    plt.xlabel("Feature importance")
    plt.savefig(RESULTS_DIR + f"feat_imp_boxplot.png", dpi=1000, bbox_inches='tight')
    plt.show()


def unimodal_performance_plot(results_single):

    df = pd.DataFrame()
    for view in ['Cellcounts_proba', 'TCR_seq_proba', 'metadata_proba', 'RNA_seq_proba']:
        print(f'Considered view: {view}')
        accuracies = []
        aucs = []
        for arr in np.split(results_single, 20, axis=0):
            arr.reset_index(drop = True, inplace = True)
            accuracies.append(balanced_accuracy_score(arr['Label'], arr[view].round()))
            aucs.append(roc_auc_score(arr['Label'], arr[view]))


        print(f'AUC: {np.mean(aucs).round(3)}+-{np.std(aucs).round(3)}')
        print(f'Accuracy: {np.mean(accuracies).round(3)}+-{np.std(accuracies).round(3)}\n\n')

        df = df.append(pd.DataFrame({'AUC' : aucs, 'Accuracy' : accuracies, 'View' : view.split('_')[0]}))


    sns.set_style("whitegrid")
    sns.set_palette("deep")

    for metric in ['AUC', 'Accuracy']:
        
        plt.figure(figsize=(8,6))
        ax = sns.swarmplot(data=df, x='View', y=metric, hue = 'View', linewidth=.5)
        sns.boxplot(data=df, x='View', y=metric, hue = 'View', dodge= False,saturation = 0, width=0.3, showfliers=False)
        ax.set(ylim=(0.3,1.0))
        plt.tight_layout()
        plt.legend([],[], frameon=False)
        plt.savefig(RESULTS_DIR + metric + "_unimodal.png", dpi=600, bbox_inches='tight')
        plt.show()

def heatmap_predproba(unimodal_proba, responder_classes, lut, row_colors):

    img = sns.clustermap(unimodal_proba[['Cellcounts_proba', 'TCR_seq_proba', 'metadata_proba', 'RNA_seq_proba']], 
                            figsize= (5,10),
                            yticklabels=responder_classes['Vaccinee'],
                            xticklabels=['Cellcounts', 'TCRseq', 'metadata', 'RNAseq'],
                            cmap="RdBu_r",
                            row_colors = row_colors,
                            dendrogram_ratio=(.1, .1),
                            cbar_pos=(-.1, .3, .03, .4),
                            )    
    for label in sorted(responder_classes['Class'].unique()):
        img.ax_col_dendrogram.bar(-0, 0, color=lut[label], label=label, linewidth=0)
        img.ax_col_dendrogram.legend(title='', bbox_to_anchor=(-.2, .3), ncol=1, frameon=False)
    plt.show()

    figure = img.fig    
    figure.savefig(RESULTS_DIR + "heatmap_singlelayers.png", dpi=600, bbox_inches='tight')