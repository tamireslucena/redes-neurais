import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def ArrumaDados(dados):

    linhas, colunas = dados.shape

    # SE CANCER DE MAMA
    # dados = dados.drop(0, axis=1)
    # print(dados)



    classes = dados[colunas-1]

    classes_numeros = pd.factorize(dados[colunas-1])[0]
    n_classes = len(set(classes_numeros))

    classes_numeros = pd.DataFrame(classes_numeros)
   
    dados = dados.drop(colunas-1, axis=1).values
    dados = StandardScaler().fit_transform(dados)    


    return dados, classes, colunas, n_classes, classes_numeros

def ReduzirDimensionalidade(dados, classes, x):

    pca = PCA(n_components=x) # nova dimensionalidade
    reducao = pca.fit_transform(dados)

    colunas = []
    for i in range(x):
        colunas.append('Dimensao ' + str(i+1))

    reducao = pd.DataFrame(data = reducao, columns = colunas)
    classes.columns = ['Classe']
    reducao = pd.concat([reducao, classes], axis = 1)
       
    variancia = pca.explained_variance_ratio_
    print(variancia)
    variancia_total = sum(variancia)
    print(variancia_total) #soma da variancia das dimensionalidades
        


    return(reducao)

def CriaGrafico(reducao, colunas, x, n_classes):

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Dimensao 1', fontsize = 15)
    ax.set_ylabel('Dimensao 2', fontsize = 15)
    ax.set_title('2 Dimensões PCA', fontsize = 20)

    colors = ['c', 'm', 'k', 'b', 'g', 'y', 'r']
    classes = []
    cores = []
        
    for i in range(n_classes):
        classes.append(i)
        cores.append(colors[i])

    for classe, cor in zip(classes, cores):
        indicesToKeep = reducao['Classe'] == classe
        ax.scatter(reducao.loc[indicesToKeep, 'Dimensao 1'], reducao.loc[indicesToKeep, 'Dimensao 2'], c = cor, s = 50)
    ax.legend(classes)
    ax.grid()
    plt.show(ax)


def main():

    dados = pd.read_csv('Glass.csv', sep=',', header=None)
    print(dados)
    dados, classes, colunas, n_classes, classes_numeros = ArrumaDados(dados)
    # print(dados)
    reducao = ReduzirDimensionalidade(dados, classes_numeros, 6)
    # CriaGrafico(reducao, colunas, 2, n_classes)

main()