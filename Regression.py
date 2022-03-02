#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 12:25:48 2021

@author: learegazzetti
"""

#Importation des packages
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

####################################################### Regression lineaire #################################################
def Regression_Lineaire(cible, expli, XTrain, XTest, YTrain, YTest, nb_cv):

    #-------------------------------------
    # Régression linéaire - régression 
    #-------------------------------------
    # Temps de calcul
    start = time.time()
    
    #Instanciation de la régression logistique
    reglin=LinearRegression()
    
    #Recherche des paramètres optimaux 
    parametres = {'fit_intercept':['True','False'], 'normalize':['True', 'False']} #Paramètres à tester    
    reglin_opt=GridSearchCV(reglin, parametres)

    #Fit sur les données d'apprentissage
    reglin_opt.fit(XTrain,YTrain)
    
    #Meilleurs parametres
    meilleurs_parametres = reglin_opt.best_params_
    
    #------------------------                  
    # Validation croisée
    # -----------------------  
    scores = cross_val_score(reglin_opt, expli, cible, cv=nb_cv, scoring='r2')
    score_moyen=np.mean(scores)
    
    #----------------------------
    #   Prédiction
    #----------------------------
    y_pred=reglin_opt.predict(XTest)
    
    #-------------------------------
    #   Affichage des métriques
    #-------------------------------
    mse=mean_squared_error(YTest,y_pred)
    
    
    #-------------------------------
    #   Graphiques
    #-------------------------------
    fig = px.scatter(x = range(len(YTest)), y = y_pred[np.argsort(YTest)], opacity=0.65, height=800, title="Predictions en bleu, données réelles en rouge")
    fig.add_traces(go.Scatter(x=np.linspace(YTest.min(), len(YTest), len(YTest)), y=np.sort(YTest), name='Données réelles', line=dict(color="#ff0000")))

    # Temps de calcul
    end = time.time()
    temps = end - start


    #-------------------------------
    #   RETURN
    #-------------------------------
    return(temps,score_moyen,mse, meilleurs_parametres, fig)



##################################################### K plus proches voisins #################################################
def K_Voisins(cible, expli, XTrain, XTest, YTrain, YTest, nb_cv):
    
    #-------------------------------------
    # K plus proches voisins - régression 
    #-------------------------------------
    #Temps de calcul
    start = time.time()
    
    #Instanciation de la régression logistique
    knn=KNeighborsRegressor()
    
    #Recherche des paramètres optimaux
    parametres = {'n_neighbors':[4,5,6],'weights':['uniform', 'distance']} #Paramètres à tester    
    knn_opt=GridSearchCV(knn, parametres)

    #Fit sur les données d'apprentissage
    knn_opt.fit(XTrain,YTrain)
    
    #Meilleurs parametres
    bestparams=knn_opt.best_params_
    #------------------------                  
    #   Validation croisée
    # -----------------------
    scores = cross_val_score(knn_opt, expli, cible, cv=nb_cv,scoring='r2')
    score_moyen=np.mean(scores)
    
    #----------------------------
    #   Prédiction
    #----------------------------
    y_pred=knn_opt.predict(XTest)
    
    #-------------------------------
    #   Affichage des métriques
    #-------------------------------
    mse=mean_squared_error(YTest,y_pred)
    
    #-------------------------------
    #   Graphiques
    #-------------------------------
    fig = px.scatter(x = range(len(YTest)), y = y_pred[np.argsort(YTest)], opacity=0.65, height=800, title="Predictions en bleu, données réelles en rouge")
    fig.add_traces(go.Scatter(x=np.linspace(YTest.min(), len(YTest), len(YTest)), y=np.sort(YTest), name='Données réelles', line=dict(color="#ff0000")))

    
    # Temps de calcul
    end = time.time()
    temps = end - start

    #-------------------------------
    #    RETURN
    #-------------------------------
    return(temps,score_moyen,mse, bestparams, fig)


##################################################### Arbre de régression #################################################
def Arbre_Regression(cible, expli, XTrain, XTest, YTrain, YTest, nb_cv):
            
    #-----------------------------------------
    # Arbre de régression - Régression
    #------------------------------------------
    #Temps de calcul
    start = time.time()
        
    #Instanciation 
    arbre_regression = DecisionTreeRegressor()
        
    #Recherche des paramètres optimaux 
    parametres = {'splitter' : ['best','random'], 'max_depth' : [5,10,15,25,30],
                      'min_samples_split':[10,15,20]}
    a_reg_opt = GridSearchCV(arbre_regression, parametres)
        
    #Fit sur les données d'apprentissage
    a_reg_opt.fit(XTrain,YTrain)
    
    #Meilleurs parametres
    meilleurs_parametres = a_reg_opt.best_params_
        
    #------------------------                  
    #   Validation croisée
    # -----------------------                 
    
    #Validation croisée
    cv_scores = cross_val_score(a_reg_opt,expli,cible,cv=nb_cv,scoring='r2')
    score_moyen = np.mean(cv_scores)
        
    #----------------------------
    #   Prédiction
    #----------------------------
    YPred = a_reg_opt.predict(XTest)
        
    #-------------------------------
    #   Affichage des métriques
    #-------------------------------
    #MSE 
    MSE = mean_squared_error(YTest, YPred)
        
    #-------------------------------
    #   Graphiques
    #-------------------------------
    fig = px.scatter(x = range(len(YTest)), y = YPred[np.argsort(YTest)], opacity=0.65, height=800, title="Predictions en bleu, données réelles en rouge")
    fig.add_traces(go.Scatter(x=np.linspace(YTest.min(), len(YTest), len(YTest)), y=np.sort(YTest), name='Données réelles', line=dict(color="#ff0000")))

    
    # Temps de calcul
    end = time.time()
    temps = end - start
        
    #-------------------------------
    #    RETURN
    #-------------------------------
    return(temps,score_moyen,MSE, meilleurs_parametres, fig)


