# miscellaneous imports
import numpy as np
import pandas as pd

# classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# training and performance analysis
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from imblearn.over_sampling import SMOTE

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA, SparsePCA

# mvlearning
from mvlearn.embed import MCCA
from mvlearn.decomposition import GroupPCA

RESULTS_DIR = './results/'


def instantiate_models(random_state = None):

    """
    Create instances of the chosen models.
        
    Returns:
        A list of tuples. The first element is the initialized model, the second its name.
    """
    
    instances = [
        (LogisticRegression(solver = 'liblinear', class_weight = 'balanced', random_state = random_state, n_jobs = -1), "LogisticRegression"),
        (RandomForestClassifier(n_estimators=100, class_weight = 'balanced', random_state = random_state, n_jobs = -1), "RandomForest"),
        ]
    return instances


def instantiate_fusions(random_state = None):
    
    """
    Create a list of data fusion implementations.
        
    Returns:
        A list of tuples. The first element is the method, the second its name.
    """
    
    MVImplementations = [
       (None,'Concat'),  ### Early integration  

       (None, 'Voting'), ### Late integration

       (SparsePCA(n_components = 3, max_iter = 100, 
                                  ridge_alpha = .05,
                                  alpha = 5,
                                  n_jobs = -1, random_state = random_state), 'PCA_singleview'), ### Early integration

       ([PCA(n_components = 3),
        PCA(n_components = 3),
        PCA(n_components = 3),
        SparsePCA(n_components = 3, max_iter = 100, 
                                  ridge_alpha = .05,
                                  alpha = 5,
                                  n_jobs = -1, random_state = random_state),], 'PCA_multiview'), ### Early integration

       (GroupPCA(
           n_components = 3,
           multiview_output=False,
           prewhiten=True, whiten=True,
           random_state=random_state
                ), 'GroupPCA'), ### Mixed/late integration

        (MCCA(n_components = 3, regs = [0.3, 
                                        0.1, 
                                        0.2, 
                                        0.9
                                       ], center = False, multiview_output=False
             ), 'MCCA'),        ### JDR (Joint Dimensionality Reduction)
    ]
    
    return MVImplementations


def mvlearn_training(multi_view, classes, implement, clf, random_state):
    
    implementation, imp_name = implement
    print(f"Implementation: {imp_name}")
    
    predictions = []
    predictions_proba = []
    ground_truth = []
    
    scaler = StandardScaler() 
    rf = LeaveOneOut()
    smote = SMOTE(sampling_strategy='minority', n_jobs=-1, random_state=random_state)

    for train_index, test_index in rf.split(multi_view[0]):
        
        
        Xs_train = [x.iloc[train_index] for x in multi_view]
        Xs_test = [x.iloc[test_index] for x in  multi_view]
        y_train, y_test = classes.iloc[train_index], classes.iloc[test_index]
        

        if imp_name == 'Concat' or imp_name == 'PCA_singleview':
            
            
            Xs_train = pd.concat(Xs_train, axis=1, ignore_index=False)
            Xs_test = pd.concat(Xs_test, axis=1, ignore_index=False)
    
            Xs_train = scaler.fit_transform(Xs_train)
            Xs_test = scaler.transform(Xs_test)
            
            Xs_train, y_train = smote.fit_resample(Xs_train, y_train)

            if imp_name == 'PCA_singleview':
                Xs_train = implementation.fit_transform(Xs_train)
                Xs_test = implementation.transform(Xs_test)
                    
        else:
            for i, _ in enumerate(Xs_train):
                scaler = StandardScaler()
                Xs_train[i] = scaler.fit_transform(Xs_train[i])
                Xs_test[i] = scaler.transform(Xs_test[i])
                Xs_train[i], new_y_train = smote.fit_resample(Xs_train[i], y_train)

            y_train = new_y_train
                
            if imp_name == 'Voting':
                pass
            
            else:
                
                if imp_name == 'PCA_multiview':

                    for i, _ in enumerate(Xs_train):
                        Xs_train[i] = implementation[i].fit_transform(Xs_train[i])
                        Xs_test[i] = implementation[i].transform(Xs_test[i])
                else:
                    Xs_train = implementation.fit_transform(Xs_train)
                    Xs_test = implementation.transform(Xs_test)
                

        if imp_name == 'PCA_multiview':
            
            Xs_train = np.hstack(Xs_train)
            Xs_test = np.hstack(Xs_test)
        
        
        if imp_name == 'Voting':
            
            temp_predictions_proba = []
            temp_predictions_proba_train = []
            
            for i, _ in enumerate(Xs_train):
                clf.fit(Xs_train[i], y_train)
                temp_predictions_proba_train.append(clf.predict_proba(Xs_train[i])[:, 1])
                temp_predictions_proba.append(clf.predict_proba(Xs_test[i])[:, 1])

            y_pred_proba = 1/len(multi_view) * sum(temp_predictions_proba)
            y_pred = np.round(y_pred_proba)
            
        else:
            
            clf.fit(Xs_train, y_train)
            y_pred_proba = clf.predict_proba(Xs_test)[:, 1]
            y_pred = clf.predict(Xs_test)


        predictions.append(y_pred)
        predictions_proba.append(y_pred_proba)
        ground_truth.append(y_test)


    predictions = [item for sublist in predictions for item in sublist]
    predictions_proba = [item for sublist in predictions_proba for item in sublist]
    ground_truth = [item for sublist in ground_truth for item in sublist]

    
    accuracy = balanced_accuracy_score(ground_truth, predictions)
    auc = roc_auc_score(ground_truth, predictions_proba, average = 'weighted')
        
    print('Balanced Accuracy: %.3f ' % accuracy)
    print('AUC: %.3f' % auc)

    return {'Balanced Accuracy': accuracy, 'AUC': auc, 'Method': imp_name}


def performance_eval(results):

    fig = make_subplots(rows=1, cols=2, horizontal_spacing = 0.01, shared_yaxes=True)

    show = True

    for i, cla in enumerate(['LogisticRegression', 'RandomForest']):
        temp = list(filter(lambda x: x['Model'] == cla, results))
        temp = [d['Results'] for d in temp]

        print(f'Model: {cla}\n')
        df = pd.DataFrame({'AUC' : list(map(lambda x: x['AUC'], temp)), 
                        'Method' : list(map(lambda x: x['Method'], temp)),
                        'Balanced Accuracy' : list(map(lambda x: x['Balanced Accuracy'], temp))})
        
        for method in df['Method'].unique():
            print(f'Method: {method}')
            print(f'AUC: {np.mean(df[df["Method"] == method]["AUC"]).round(3)}+-{np.std(df[df["Method"] == method]["AUC"]).round(3)}')
            print(f'Accuracy: {np.mean(df[df["Method"] == method]["Balanced Accuracy"]).round(3)}+-{np.std(df[df["Method"] == method]["Balanced Accuracy"]).round(3)}\n\n')
        
        fig.add_trace(go.Box(
            name='Accuracy',
            x=list(map(lambda x: x['Method'], temp)),
            y=list(map(lambda x: x['Balanced Accuracy'], temp)),
            text=list(map(lambda x: np.around(x['Balanced Accuracy'], decimals = 2), temp)),
            marker_color='steelblue',
            showlegend = show,
            boxpoints='all',
            jitter=0.1,
            legendgroup=cla,
                    ),row=1, col=i+1)
        
        fig.add_trace(go.Box(
            name='AUC',
            x=list(map(lambda x: x['Method'], temp)),
            y=list(map(lambda x: x['AUC'], temp)),
            text=list(map(lambda x: np.around(x['AUC'], decimals = 2), temp)),
            marker_color='firebrick',
            showlegend = show,
            boxpoints='all',
            jitter=0.1,
                    ),row=1, col=i+1)
        show = False
    

    fig.update_yaxes(gridcolor='grey', gridwidth = .1, title_text="", range=[0.19, 1.01])
    fig.update_xaxes(zerolinewidth=1, zerolinecolor='grey')

    fig.update_layout(
        font=dict( size=16, color="dark grey"),
            legend_title="", title_text="", 
            height=600, width=1500,
            plot_bgcolor='rgb(255, 255, 255)', 
            boxmode='group', boxgap=0.25, boxgroupgap=0.00
        )

    fig.write_image(RESULTS_DIR + 'performance_boxplots.png', height=600, width=1000, scale=6)
    fig.show()


def single_view_training(Xs, responder_classes):


    rf = LeaveOneOut()
    results_single = []
    coefficients = []

    classi = np.array([1 if cla == 'Early-converter' else 0 for cla in responder_classes])

    for random_state in range(42, 62):
        smote = SMOTE(sampling_strategy='minority', n_jobs=-1, random_state=random_state)
        scaler = StandardScaler()
        lr = LogisticRegression(solver = 'liblinear',
                                class_weight = 'balanced', random_state = random_state, n_jobs = -1)

        for train_index, test_index in rf.split(Xs[0]):

            Xs_train = [x.iloc[train_index] for x in Xs]
            Xs_test = [x.iloc[test_index] for x in  Xs]
            y_train, y_test = classi[train_index], classi[test_index]


            for i,_ in enumerate(Xs_train):
                
                Xs_train[i], new_y_train = smote.fit_resample(Xs_train[i], y_train)
                Xs_train[i] = scaler.fit_transform(Xs_train[i])
                Xs_test[i] = scaler.transform(Xs_test[i])


            predictions_proba = []

            for i, _ in enumerate(Xs_train):
                lr.fit(Xs_train[i], new_y_train)
                coefficients.append(lr.coef_)
                predictions_proba.append(lr.predict_proba(Xs_test[i])[:, 1])

            results_single.append({'Cellcounts_proba': float(predictions_proba[0].round(3)), 
                                    'TCR_seq_proba': float(predictions_proba[1].round(3)), 
                                    'metadata_proba': float(predictions_proba[2].round(3)), 
                                    'RNA_seq_proba': float(predictions_proba[3].round(3)),
                                    'Label': int(y_test)})
            
    return coefficients, pd.DataFrame(results_single)