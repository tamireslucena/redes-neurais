import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def NormalizaDados(dados):
    normal = MinMaxScaler()
    normalizados = normal.fit_transform(dados)
    normalizados = pd.DataFrame(normalizados)
    indices = dados.index
    normalizados.index = indices
    return normalizados

def ArrumaDados(dados):

    # # se cancer de mama
    # dados = dados.drop(columns=0)

    linhas, colunas = dados.shape

    classificados = pd.factorize(dados[colunas-1])[0]
    n_classes = len(set(classificados))

    
    saida_esperada = []
    for classe in classificados:
        aux = [1 if classe == i else 0 for i in range(n_classes)]
        saida_esperada.append(aux)
    saida_esperada = np.array(saida_esperada)
    
    dados = dados.drop(colunas-1, axis=1) # remove coluna de classificacao
    dados = NormalizaDados(dados)
    dados[colunas-1] = 1 # adiciona bias
    dados = dados.values

    print(n_classes)
    return dados, saida_esperada, n_classes, linhas, colunas

def InicializaPesos(n_camadas, n_entradas, n_perceptrons):
    pesos = []
    for c in range(n_camadas):
        if(c==0): aux = [np.array([random.uniform(0, 0.1) for i in range(n_entradas)]) for p in range(n_perceptrons[c])]
        else: aux = [np.array([random.uniform(0, 0.1) for i in range(n_perceptrons[c-1]+1)]) for p in range(n_perceptrons[c])]
        aux = np.array(aux)
        pesos.append(aux)
    pesos = np.array(pesos)

    # print(pesos)
    return pesos

def Ativacao(u):
    return 1.0/(1.0+math.exp(-u))

def Derivada(u):
    #return (math.exp(-u))/((1+math.exp(-u))**2)
    return u * (1-u)

def Acuracia(saida, rotulos):
    saida = [np.argmax(s) for s in saida]
    rotulo = [np.argmax(r) for r in rotulos]
    
    acuracia = [1 if saida[i] == rotulo[i] else 0 for i in range(len(saida))]
    acuracia = sum(acuracia)/float(len(acuracia))

    return acuracia

def CriaGrafico(epocas, erros_treino, erros_teste, acuracia):

    plt.subplot(211)
    plt.plot(epocas, erros_treino, label='treino')
    plt.plot(epocas, erros_teste, label='teste')
    plt.xlabel("Épocas")
    plt.ylabel("Erro")
    plt.legend()

    plt.subplot(212)
    plt.plot(epocas, acuracia, label='acuracia')
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.show()

def main(n_camadas=3, taxa_erro=0.1, eta=0.5):
    
    dados = pd.read_csv('Glass.csv', header=None)

    dados, saida_esperada, n_classes, n_linhas, n_colunas= ArrumaDados(dados)
    

    treino, teste, rotulos_treino, rotulos_teste = train_test_split(dados, saida_esperada, test_size=0.2, random_state=50)

    n_entradas_treino = len(treino[0])
    
    n_perceptrons = [2,3,6]
    
    for i in range(n_camadas-1):
        n_perceptrons.append(n_colunas)
    n_perceptrons.append(n_classes)
    print(n_perceptrons)

    
    random.seed(50)
    pesos = InicializaPesos(n_camadas, n_entradas_treino, n_perceptrons)    
    epoca = 0
    epocas = []


    erros_treino = []
    erros_teste = []
    
    acuracia = []

    e = 1

    while(e > taxa_erro):

        erro_epoca = []

        ############################################# TREINO #############################################
    
        for i in range(len(treino)):


            erros_saida_treino = []

            # IDA
            saida_ativacao_treino = []
            for c in range(n_camadas):
                saida_ativacao_treino.append([])

            deltas = []
            for c in range(n_camadas):
                deltas.append([])


            
            # IDA
            for c in range(n_camadas):

                # primeira camada recebe entradas
                if(c == 0):
                    # print('treino', c)
                    for p in range(n_perceptrons[c]):
                        aux = Ativacao(np.dot(treino[i], pesos[c][p]))
                        saida_ativacao_treino[c].append(aux)
                    saida_ativacao_treino[c].append(1)

                # ultima camada n adiciona bias
                elif(c == n_camadas-1):
                    
                    # print('treino', c)
                    for p in range(n_perceptrons[c]):
                        
                        aux = Ativacao(np.dot(saida_ativacao_treino[c-1], pesos[c][p]))
                        saida_ativacao_treino[c].append(aux)

                else:
                    
                    # print('treino', c)
                    for p in range(n_perceptrons[c]):
                        aux = Ativacao(np.dot(saida_ativacao_treino[c-1], pesos[c][p]))
                        saida_ativacao_treino[c].append(aux)
                    saida_ativacao_treino[c].append(1)

            # VOLTA
            for c in range(n_camadas-1, -1, -1):

                # ultima camada
                if(c == n_camadas-1):
                    for p in range(n_perceptrons[c]):
                        erro = rotulos_treino[i][p] - saida_ativacao_treino[c][p]
                        erros_saida_treino.append(erro)
                        
                        derivada = Derivada(saida_ativacao_treino[c][p])
                        deltas[c].append(derivada * erro)

                else:
                    for p in range(n_perceptrons[c]):
                        aux = 0
                        for p2 in range(n_perceptrons[c+1]):
                            aux += pesos[c+1][p2][p] * deltas[c+1][p2]
                            derivada = Derivada(saida_ativacao_treino[c][p])                            
                        deltas[c].append(derivada * aux)

            err_treino = []
            for er in erros_saida_treino:
                err_treino.append(er**2)
            media_erro_treino = np.mean(err_treino)
            erro_epoca.append(media_erro_treino)
            
            if(media_erro_treino > taxa_erro):

                
                # atualiza pesos
                for c in range(n_camadas-1, -1, -1):
                    for p in range(n_perceptrons[c]):

                        #primeira camada
                        if(c == 0):
                            for pe in range(len(pesos[c][p])):
                                pesos[c][p][pe] += (eta*treino[i][pe]*deltas[c][p])

                        else:
                            for pe in range(len(pesos[c][p])):
                                pesos[c][p][pe] += (eta*saida_ativacao_treino[c-1][pe]*deltas[c][p])
                                                   

        # média de erros
        e = np.mean(erro_epoca)
        erros_treino.append(e)
        
        print("\nÉpoca ", epoca)
        print("Erro de treino da época: ", round(e, 5))


        epoca+=1
        epocas.append(epoca)




        ############################################# TESTE #############################################

        teste_estimado = []
        erro_quadr_saida = []

        for i in range(len(teste)):
            
            saida_ativacao_teste = []
            for c in range(n_camadas):
                saida_ativacao_teste.append([])

            for c in range(n_camadas):   

                # primeira camada
                if(c == 0):
                    
                    # print('teste', c)
                    for p in range(n_perceptrons[c]):
                        aux = Ativacao(np.dot(teste[i], pesos[c][p]))
                        saida_ativacao_teste[c].append(aux)
                    saida_ativacao_teste[c].append(1)
                
               # ultima camada n adiciona bias
                elif(c == n_camadas-1):
                    
                    # print('teste', c)
                    for p in range(n_perceptrons[c]):                        
                        aux = Ativacao(np.dot(saida_ativacao_teste[c-1], pesos[c][p]))
                        saida_ativacao_teste[c].append(aux)

                else:
                    
                    # print('teste', c)
                    for p in range(n_perceptrons[c]):
                        aux = Ativacao(np.dot(saida_ativacao_teste[c-1], pesos[c][p]))
                        saida_ativacao_teste[c].append(aux)
                    saida_ativacao_teste[c].append(1)
        

            aux_erros = []
            for p in range(n_perceptrons[n_camadas-1]):
                aux_erros.append(rotulos_teste[i][p] - saida_ativacao_teste[n_camadas-1][p])

            aux = [e**2 for e in aux_erros]
            result = np.mean(aux)
            erro_quadr_saida.append(result)


            teste_estimado.append(saida_ativacao_teste[n_camadas-1])
        
        print('Erro: ',np.mean([e**2 for e in aux_erros]))
        erros_teste.append(np.mean(erro_quadr_saida))

        a = Acuracia(teste_estimado, rotulos_teste)
        acuracia.append(a)

        print('Acuracia: ', round(a, 5))
        print('e: ', e)
    
   

    CriaGrafico(epocas, erros_treino, erros_teste, acuracia)

    return



main()