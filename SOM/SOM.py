import pandas as pd
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def NormalizaDados(dados):
    normal = MinMaxScaler()
    normalizados = normal.fit_transform(dados)
    normalizados = pd.DataFrame(normalizados)
    indices = dados.index
    normalizados.index = indices
    return normalizados

def ArrumaDados(dados):
    linhas, colunas = dados.shape # 150, 5

    dados = dados.sample(frac=1, random_state=1) #randomiza linhas
    dados = dados.reset_index() #novos indices
    dados = dados.drop(columns=['index']) #remove indices antigos
    
    classes = dados[colunas-1]
    classes = pd.factorize(classes)[0]
    
    dados = dados.drop(colunas-1, axis=1)    
    dados = NormalizaDados(dados)
    dados = dados.values

    return dados, linhas, colunas, classes

def ArrumaDadosWine(dados):
    linhas, colunas = dados.shape 
    classes = dados[0]

    dados = dados.sample(frac=1) #randomiza linhas
    dados = dados.reset_index() #novos indices
    dados = dados.drop(columns=['index']) #remove indices antigos

    dados = dados.drop(columns=0)

    dados = NormalizaDados(dados)
    dados = dados.values

    return dados, linhas, colunas

def criaGrafico(rede, linhas, colunas):

    aux_grafico = np.zeros((linhas, colunas))
    for l in range(linhas):
        for c in range(colunas):
            aux_grafico[l][c] = math.sqrt(sum([a**2 for a in rede[l][c]]))
    plt.imshow(aux_grafico)
    plt.show()

def InicializaGrid(dados, linhas):
    dimensao = math.sqrt(2)*linhas
    dimensao = math.sqrt(dimensao)
    dimensao = int(dimensao)
    qtd_atributos = len(dados[0])
    
    # Inicializa pesos aleatorios
    grid = np.random.uniform(low=-0.1, high=0.1, size=(dimensao, dimensao, qtd_atributos))

    return grid, qtd_atributos, dimensao

def EncontraIndiceGrid(indice_vencedor, dimensao):
    linha_min = (indice_vencedor // dimensao)
    coluna_min = (indice_vencedor % dimensao)
    return linha_min, coluna_min

def Distancia (a, b):
    return np.sqrt(np.sum([(a[i]-b[i])**2 for i in range(len(a))]))

def CalculaDistancias(dados, grid, linhas, qtd_atributos, dimensao):
    distancias = []

    for l in range(dimensao):
        for c in range(dimensao):
            distancias.append(Distancia(dados, grid[l][c]))
    indice_vencedor = np.argmin(distancias)
    linha_min, coluna_min = EncontraIndiceGrid(indice_vencedor, dimensao)

    return distancias, indice_vencedor, linha_min, coluna_min  

def CalculaDistanciaVizinhos(dimensao, linha_min, coluna_min):
    distancia_vizinhos = []
    for l in range(dimensao):
        for c in range(dimensao):
            distancia_vizinhos.append(Distancia([linha_min, coluna_min], [l, c]))
        
    return distancia_vizinhos

def AtualizaPesos(grid, h, dimensao, eta, dados, i, qtd_atributos):
    for l in range(dimensao):
        for c in range(dimensao):
            for a in range(qtd_atributos):
                grid[l][c][a] += eta * (dados[i][a]-grid[l][c][a]) * h[l][c]
    return grid
    
def RotulaNeuronio(grid, dados, dimensao, classes):
    menores = []

    for l in range(dimensao):
        for c in range(dimensao):
            distancias = []
            for i in range(len(dados)):
                distancias.append(Distancia(grid[l][c], dados[i]))
            indice = np.argmin(distancias)
            menores.append(classes[indice])

    return menores

def Umatrix(grid, dados, dimensao, classes):

    matriz = np.zeros((dimensao, dimensao)) #cria matriz iniciadas com zeros

    for l in range(dimensao):
        for c in range(dimensao):
        
            if(l == 0 and c == 0):
                matriz[l][c] = (Distancia(grid[l][c], grid[l][c+1]) + Distancia(grid[l][c], grid[l+1][c]) + Distancia(grid[l][c], grid[l+1][c+1]))/3

            elif(l == dimensao-1 and c == 0):
                matriz[l][c] = (Distancia(grid[l][c], grid[l][c+1]) + Distancia(grid[l][c], grid[l-1][c]) + Distancia(grid[l][c], grid[l-1][c+1]))/3
            
            elif(l == 0 and c == dimensao-1):
                matriz[l][c] = (Distancia(grid[l][c], grid[l][c-1]) + Distancia(grid[l][c], grid[l+1][c]) + Distancia(grid[l][c], grid[l+1][c-1]))/3

            elif(l == dimensao-1 and c == dimensao-1):
                matriz[l][c] = (Distancia(grid[l][c], grid[l][c-1]) + Distancia(grid[l][c], grid[l-1][c]) + Distancia(grid[l][c], grid[l-1][c-1]))/3

            elif(l == dimensao-1 and c != 0 and c != dimensao-1):
                matriz[l][c] = (Distancia(grid[l][c], grid[l-1][c-1]) + Distancia(grid[l][c], grid[l-1][c]) + Distancia(grid[l][c], grid[l-1][c+1]) + Distancia(grid[l][c], grid[l][c-1]) + Distancia(grid[l][c], grid[l][c+1]))/5
            
            elif(l == 0 and c != 0 and c != dimensao-1):
                matriz[l][c] = (Distancia(grid[l][c], grid[l][c-1]) + Distancia(grid[l][c], grid[l][c+1]) + Distancia(grid[l][c], grid[l+1][c-1]) + Distancia(grid[l][c], grid[l+1][c]) + Distancia(grid[l][c], grid[l+1][c+1]))/5

            elif(l != 0 and l != dimensao-1 and c == dimensao-1):
                matriz[l][c] = (Distancia(grid[l][c], grid[l-1][c-1]) + Distancia(grid[l][c], grid[l-1][c]) + Distancia(grid[l][c], grid[l][c-1]) + Distancia(grid[l][c], grid[l+1][c-1]) + Distancia(grid[l][c], grid[l+1][c]))/5

            elif(l != 0 and l != dimensao-1 and c == 0):
                matriz[l][c] = (Distancia(grid[l][c], grid[l-1][c]) + Distancia(grid[l][c], grid[l-1][c+1]) + Distancia(grid[l][c], grid[l][c+1]) + Distancia(grid[l][c], grid[l+1][c]) + Distancia(grid[l][c], grid[l+1][c+1]))/5

            else:
                matriz[l][c] = (Distancia(grid[l][c], grid[l-1][c-1]) + Distancia(grid[l][c], grid[l-1][c]) + Distancia(grid[l][c], grid[l-1][c+1]) + Distancia(grid[l][c], grid[l][c-1]) + Distancia(grid[l][c], grid[l][c+1]) + Distancia(grid[l][c], grid[l+1][c-1]) + Distancia(grid[l][c], grid[l+1][c]) + Distancia(grid[l][c], grid[l+1][c+1]))/8
    
    menores = RotulaNeuronio(grid, dados, dimensao, classes)

    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    im = ax.imshow(matriz, interpolation='bilinear')
    
    for l in range(dimensao):
        for c in range(dimensao):
            ax.text(c, l, menores[(dimensao*l)+c], color='black', ha='center', va='center')
    
    fig.colorbar(im)
    plt.show()
    


def main():

    dados = dados2 = pd.read_csv('Taiacupeba_Variaveis.csv', sep=',', header=None)
    dados, linhas, colunas, classes = ArrumaDados(dados)
    print(dados)
    
    grid, qtd_atributos, dimensao = InicializaGrid(dados, linhas)

    max_epocas = 100
    eta0 = eta = 0.5
    sigma0 = sigma = None
    

    for e in range(max_epocas):

        #print("Época: ", e, "- ALFA: ", eta, " - SIGMA: ", sigma)

        for i in range(linhas):
            
            distancias, indice_vencedor, linha_min, coluna_min = CalculaDistancias(dados[i], grid, linhas, qtd_atributos, dimensao)
            
            d = CalculaDistanciaVizinhos(dimensao, linha_min, coluna_min)

            if(sigma is None):
                sigma = sigma0 = math.sqrt(-(dimensao**2) / (2*math.log(0.1)))
                tau = max_epocas/np.log(sigma0/0.1)

            h = [np.exp((-d[j]**2)/(2*sigma**2)) for j in range(len(d))]
            h = np.reshape(h, (dimensao, dimensao))
            

            grid = AtualizaPesos(grid, h, dimensao, eta, dados, i, qtd_atributos)
    
        
        eta = eta0 * np.exp(-e/tau)
        #sigma = sigma0 * np.exp(-e/tau)
        
    criaGrafico(grid, dimensao, dimensao)
    Umatrix(grid, dados, dimensao, classes)

main()