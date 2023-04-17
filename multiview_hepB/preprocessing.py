import pandas as pd
import re
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from natsort import natsorted
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

'''
Credit to pydeseq2: https://github.com/owkin/PyDESeq2
'''
from multiview_hepB.py_deseq import py_DESeq2


def readcounts_preprocessing(readcounts, responder_classes, gene_mapping):

    """EXP0 are the samples prior to vaccination, the others are subsequent (EXP3 = day 3, EXP7 = day 7).
    Each individual has a replicate (1/2). Two columns for each person are present.
    A person is always identified by H plus a number.
    S plus a number identified the blood sample id.
    One column is kept for each vaccinee holding the sum of the replicates.
    """
    genenames = readcounts['genename']
    readcounts = readcounts.filter(regex='EXP0')
    colnames = [x.split('_')[0] for x in list(readcounts.columns)][::2]
    i = 2
    for x in range(0, len(readcounts.columns), i):
        readcounts.iloc[:, x] = readcounts.iloc[:, x:x+i].sum(axis=1)
        readcounts = readcounts.rename(columns={readcounts.columns[x]:f'sum{x // i + 1}'})
    readcounts = readcounts.filter(regex='^sum')

    readcounts = readcounts[readcounts.ne(0).sum(1).ge(10)]
    readcounts.columns = colnames
    # Drop non-converters
    readcounts.drop(columns=[col for col in readcounts if col not in list(responder_classes['Vaccinee'])], inplace = True)

    readcounts['Gene.Name'] = genenames
    readcounts = readcounts[readcounts['Gene.Name'].str.contains('_') ==  False]
    readcounts["Gene.Name"] = readcounts["Gene.Name"].map(dict(zip(gene_mapping["Gene.ID"], gene_mapping["Gene.Name"])))

    ### Gender related genes previously identified are removed
    genes_to_remove = ["Xist_exon4", "Xist_exon1",
                        "XIST", "TSIX", "EIF1AY", "ZFY",
                        "DDX3Y", "RPS4Y1", "TXLNGY", "USP9Y",
                        "UTY", "KDM5D", "PRKY"]

    readcounts = readcounts[~readcounts['Gene.Name'].isin(genes_to_remove)]
    # Hemoglobin subunits are removed
    readcounts = readcounts[~readcounts['Gene.Name'].str.startswith('HB')]

    # Genes with less than 100 total transcripts are removed
    readcounts = readcounts.loc[readcounts.sum(axis=1) > 100]
    readcounts.reset_index(drop = True, inplace = True)

    print(f'Readcounts size: {readcounts.shape}')
    return readcounts


def deseq2(readcounts, responder_classes):
    """Gene counts normalization through DESeq2.
    """
    rpy2_logger.setLevel(logging.ERROR)
    dds = py_DESeq2(count_matrix = readcounts.reindex(natsorted(readcounts.columns), axis=1),
                    design_matrix = responder_classes.set_index('Vaccinee'),
                design_formula = '~Class',
                gene_column = 'Gene.Name')
    
    dds.run_deseq() 
    dds.get_deseq_result(contrast=["Class", "Late-converter", "Early-converter"])
    res = dds.deseq_result 
    normalized_count_hep = round(dds.normalized_count())

    return res, normalized_count_hep


def aggregate_modules(typ, normalized_count_hep, module_list):

    """Aggregate RNA-seq data in modules based on chosen operator. Multiple aggregators are implemented:
    -'mean'
    -'median'
    -'std'
    -'pca'
    """
    
    normalized_count_hep['Gene.Name'] = [x.upper() for x in normalized_count_hep['Gene.Name']]
    normalized_count_hep.replace({"Gene.Name": dict(zip(module_list['Gene'], module_list['Module'] + '.' + module_list['Function']))}, inplace=True)
    modules = [x for x in normalized_count_hep['Gene.Name'].unique() if re.match('M[0-9]{1,3}\.[0-9]{1,3}', x)]
    
    
    # Keep only data present in the models and clinical data
    normalized_count_hep = normalized_count_hep.loc[normalized_count_hep['Gene.Name'].isin(modules)]

    if typ.startswith('pca') | typ.endswith('pca'):
        
        pca = PCA(n_components=1)
        scaler = StandardScaler()
        agg = normalized_count_hep.groupby('genename', as_index = False)

        aggregated = pd.DataFrame()
        for group_name, df_group in agg:
            df_group = pca.fit_transform(scaler.fit_transform(df_group.drop(columns='genename').to_numpy().T)).T
            aggregated[group_name] = df_group.flatten().tolist()

        return aggregated

    if typ.startswith('mean') | typ.endswith('mean'):
        agg = normalized_count_hep.groupby('Gene.Name', as_index = False).mean().round(2)   
    elif typ.startswith('median') | typ.endswith('median'):
        agg = normalized_count_hep.groupby('Gene.Name', as_index = False).median().round(2)
    elif typ.startswith('std') | typ.endswith('std'):
        agg = normalized_count_hep.groupby('Gene.Name', as_index = False).std(ddof=0).round(2)


    # Transpose for the final matrix
    pose = agg.T
    pose.columns = pose.iloc[0]
    pose = pose.iloc[1:]
    pose.reset_index(drop = True, inplace = True)

    aggregated = pose.round(0).astype(int)
    return aggregated