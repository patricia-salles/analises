# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 23:10:32 2019

@author: pati_
"""

cd C:\Users\pati_\Desktop\analises\etl

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
df=pd.read_csv("pagamentos.csv", sep=",")

'''
transformar objeto de data em datetime
coluna fase 3 valores - EMPENHO, PAGAMENTO, LIQUIDAÇÃO
coluna credor - muitos 8000 valores - empresa ou pessoa
coluna valor - muitos 123400 - transformar para número - tá com $
coluna numero - muitos 190000
coluna cpf_cnpj - muitos 7000
transformar objeto de data em datetime
coluna numero_processo- - muitos 18000
coluna bem_ou_servico_prestado - muitos 150000
coluna natureza é um agrupamento de bem_ou_servico_prestado - 800
coluna acao não sei o que é, mas parece ser um tipo de agrupamento de bem_ou_servico_prestado - 1000
coluna funcao - definidos no tesouro nacional - 50
coluna subfuncao - agrupamento da funcao - 76
coluna processo_licitatorio - tem uns valores escritos errados - 27 ajustar dicionário
coluna fonte_recurso - tem uns valores escritos errados acho - 40 ajustar dicionário

'''
# Colunas para datetime
#Cols = ['data_publicacao', 'data_pagamento']
#df[Cols]=pd.to_datetime(df[Cols], dayfirst=True)
df.data_pagamento=pd.to_datetime(df.data_pagamento, dayfirst=True)

# Remover R$
df['valor'] = df['valor'].str[3:]
# Remover espaço em branco
df['valor']=df['valor'].str.replace(" ", "")
# Pensar em tirar os valores #### e ,00 para analisar sem eles e depois olhar só esses caras

# Dicionário colunas com valores erro grafia
df['processo_licitatorio'] = df['processo_licitatorio'].str.upper().

dict = {
        'EXIGIBILIDADE': 'INEXIGIBILIDADE',
        'NEXIBILIDADE': 'INEXIGIBILIDADE',
        'XIBILIDADE': 'INEXIGIBILIDADE',
        'NEXIGIBILIDADE': 'INEXIGIBILIDADE',
        'INEXIBILIDADE': 'INEXIGIBILIDADE',
        'XIGIBILIDADE': 'INEXIGIBILIDADE',        
        'SPENSA': 'DISPENSA',
        'ISPENSA': 'DISPENSA',
        'PENSA': 'DISPENSA', 
        'SENTO': 'ISENTO',
        'ENTO': 'ISENTO',
        'REGAO': 'PREGAO',
        'EGAO': 'PREGAO',
        'OMADA DE PRECO': 'TOMADA DE PRECO',
        'ONCORRENCIA': 'CONCORRENCIA',
        'NCORRENCIA' : 'CONCORRENCIA'}

df.replace({'processo_licitatorio': dict}, inplace = True)

'''
coluna fonte recurso
array(['0001 - REC.IMP.TRANSF.EDUCACAO 25%', '0000 - RECURSOS ORDINARIOS',
       '0050 - REC.PROPRIAS ENT.ADM.', '0015 - TRANSFERENCIA FNDE',
       '0003 - CONT.REGIME PRPPRIO PREV.SOCIA', '0014 - TRANSF.REC. SUS',
       '0002 - REC.IMP.TRANSF.IMP.SAUDE 15%', '0000 - TESOURO',
       '0029 - TRANSF. REC. - FNAS',
       '0019 - TRANSF.FUNDEB - OUT. DESPESAS',
       '0024 - TRANSF. CONV. OUTROS', '0016 - CONTRIBUICAO - CIDE',
       '0018 - TRANSF. FUNDEB PESSOAL', '0030 - TRANSF. FIES',
       '0022 - TRANSF. CONVENIO EDUCACAO',
       '0004 - CONT.PROG. SAL. EDUCACAO',
       '0023 - TRANSF.CONVENIOS - SAUDE', '0024 - TRANSF. DO ESTADO',
       '0042 - ROYALTIES/FUNDO ESP.PET.REC.MI',
       '0050 - REC.PROPRIAS ENT.ADM.INDIRETAS', nan,
       '0014 - TRANSF. REC. SUS', '0024 - Transferências de Convênios',
       '000 - TESOURO                                                                                                                                        1',
       '00 - TESOURO                                                                                                                                        14',
       '00 - RECURSOS ORDINARIOS                                                                                                                            15',
       '050 - REC.PROPRIAS ENT.ADM.INDIRETAS                                                                                                                 1',
       '000 - RECURSOS ORDINARIOS                                                                                                                            1',
       '001 - REC.IMP.TRANSF.EDUCACAO 25%                                                                                                                    1',
       '00 - TESOURO                                                                                                                                        15',
       '0090 - OPERACOES DE CREDITOS INTERNAS',
       '29 - TRANSF. REC. - FNAS                                                                                                                            17',
       '00 - RECURSOS ORDINARIOS                                                                                                                            17',
       '0003 - CONT.REGIME PROPRIO PREV.SOCIAL',
       '4 - TRANSF.REC. SUS                                                                                                                                170',
       '029 - TRANSF. REC. - FNAS                                                                                                                            1',
       '0 - RECURSOS ORDINARIOS                                                                                                                            180',
       '01 - REC.IMP.TRANSF.EDUCACAO 25%                                                                                                                    18',
       '00 - RECURSOS ORDINARIOS                                                                                                                            18',
       '01 - REC.IMP.TRANSF.EDUCACAO 25%                                                                                                                    19'],
      dtype=object)


# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

plotPerColumnDistribution(df, 10, 5)


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    #filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()
    
plotCorrelationMatrix(df, 2)

# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()
    
plotScatterMatrix(df, 2, 10)