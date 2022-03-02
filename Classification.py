#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 11:24:18 2021

@author: learegazzetti
"""

#Importation des packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
import plotly.express as px
import plotly.graph_objects as go
import time


####################################################### Arbre de décision #################################################
def Arbre_Decision(cible, expli, XTrain, XTest, YTrain, YTest, nb_cv):
    
    #-------------------------------------
    # Arbre de décision - classification 
    #-------------------------------------
    # Temps de calcul
    start = time.time()
    
    #Instanciation 
    dtree = DecisionTreeClassifier()
    
    #Recherche des paramètres optimaux 
    parametres = {'criterion' : ['gini','entropy'], 'splitter' : ['best','random'],
                  'max_depth' : [5,10,15], 'min_samples_split' : [2,5,10,15]}
    dtree_opt = GridSearchCV(dtree, parametres)
    
    #Fit sur les données d'apprentissage
    dtree_opt.fit(XTrain,YTrain)
    
    #Meilleurs parametres
    meilleurs_parametres = dtree_opt.best_params_
    
    #------------------------                  
    # Validation croisée
    # -----------------------                 
    cv_scores = cross_val_score(dtree_opt,expli,cible,cv=nb_cv,scoring='accuracy')
    score_moyen = np.mean(cv_scores)
    
    #----------------------------
    #   Prédiction
    #----------------------------
    YPred = dtree_opt.predict(XTest)
    
    #-------------------------------
    #   Affichage des métriques
    #-------------------------------
    #Matrice de confusion 
    matrice_confusion = metrics.confusion_matrix(YTest,YPred)
    
    #Rappel et précision
    if len(np.unique(cible))>2:
        precision = precision_score(YTest,YPred,average='weighted')
        rappel = recall_score(YTest,YPred,average='weighted')
    else :
        precision = precision_score(YTest,YPred,pos_label=np.unique(cible)[1])
        rappel = recall_score(YTest,YPred,pos_label=np.unique(cible)[1])

    #-------------------------------
    #   Graphiques
    #-------------------------------
    #Courbe ROC 
    #Calcul de la probabilité d'appartenance à chaque classe
    proba_positif =  dtree_opt.predict_proba(XTest)

    if (len(np.unique(cible)) > 2):
        # Affichage courbe ROC
        y_onehot = pd.get_dummies(YTest, columns=dtree_opt.classes_)
    
        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1 )

        for i in range(proba_positif.shape[1]):
            y_true = y_onehot.iloc[:, i]
            y_score = proba_positif[:, i]

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)

            name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

        fig.update_layout(
            xaxis_title='Taux de faux positifs',
            yaxis_title='Taux de vrais positifs',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700, height=700)

    else : 
        fp, vp, _ = roc_curve(YTest, proba_positif[:,1],pos_label=np.unique(cible)[1])
        
        #calcul de l'aire sous la courbe
        aire_courbe = roc_auc_score(YTest, proba_positif[:,1])
        
        # Affichage courbe ROC
        fig = px.area(
            x=fp, y=vp,
            title=f'Courbe ROC modalité={np.unique(cible)[1]} (AUC={aire_courbe:.4f})',
            labels=dict(
                x='Taux de faux positifs', 
                y='Taux de vrais positifs'), width=700, height=700)
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1)
        
    
    # Temps de calcul
    end = time.time()
    temps = end - start
    
    #-------------------------------
    #   RETURN
    #-------------------------------
    return(score_moyen,matrice_confusion,temps, meilleurs_parametres,precision,rappel, fig)




####################################################### Analyse discriminante #################################################
def Analyse_Discriminante(cible, expli, XTrain, XTest, YTrain, YTest, nb_cv):

    #-------------------------------------
    # Analyse discriminante - classification 
    #-------------------------------------
    #Temps de calcul
    start = time.time()
    
    #Instanciation 
    a_discri = LinearDiscriminantAnalysis()
        
    #Recherche des paramètres optimaux 
    parametres = {'solver' : ['svd','lsqr','eigen']}
    a_discri_opt = GridSearchCV(a_discri, parametres)
        
    #Fit sur les données d'apprentissage
    a_discri_opt.fit(XTrain,YTrain)
    
    #Meilleurs parametres
    meilleurs_parametres = a_discri_opt.best_params_
        
    #------------------------                  
    #   Validation croisée
    # ----------------------- 
    cv_scores = cross_val_score(a_discri_opt,expli,cible,cv=nb_cv,scoring='accuracy')
    score_moyen = np.mean(cv_scores)
        
    #----------------------------
    #   Prédiction
    #----------------------------
    YPred = a_discri_opt.predict(XTest)
        
    #-------------------------------
    #   Affichage des métriques
    #-------------------------------
    #Matrice de confusion 
    matrice_confusion = metrics.confusion_matrix(YTest,YPred)
    
    #Rappel et précision
    if len(np.unique(cible))>2:
        precision = precision_score(YTest,YPred,average='weighted')
        rappel = recall_score(YTest,YPred,average='weighted')
    else :
        precision = precision_score(YTest,YPred,pos_label=np.unique(cible)[1])
        rappel = recall_score(YTest,YPred,pos_label=np.unique(cible)[1])
    
    #-------------------------------
    #   Graphiques
    #-------------------------------
    #Courbe ROC 
    #Calcul de la probabilité d'appartenance à chaque classe
    proba_positif =  a_discri_opt.predict_proba(XTest)

    if (len(np.unique(cible)) > 2):
        # Affichage courbe ROC
        y_onehot = pd.get_dummies(YTest, columns=a_discri_opt.classes_)
    
        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1 )

        for i in range(proba_positif.shape[1]):
            y_true = y_onehot.iloc[:, i]
            y_score = proba_positif[:, i]

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)

            name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

        fig.update_layout(
            title='Courbe ROC',
            xaxis_title='Taux de faux positifs',
            yaxis_title='Taux de vrais positifs',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700, height=700)

    else : 
        fp, vp, _ = roc_curve(YTest, proba_positif[:,1],pos_label=np.unique(cible)[1])
        
        #calcul de l'aire sous la courbe
        aire_courbe = roc_auc_score(YTest, proba_positif[:,1])
        
        # Affichage courbe ROC
        fig = px.area(
            x=fp, y=vp,
            title=f'Courbe ROC modalité={np.unique(cible)[1]} (AUC={aire_courbe:.4f})',
            labels=dict(
                x='Taux de faux positifs', 
                y='Taux de vrais positifs'), width=700, height=700)
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1)

    # Temps de calcul
    end = time.time()
    temps = end - start
    
    #-------------------------------
    #    RETURN
    #-------------------------------
    return(score_moyen,matrice_confusion,temps,meilleurs_parametres, precision, rappel, fig)
        



####################################################### Régression logistique #################################################
def Regression_Logistique(cible, expli, XTrain, XTest, YTrain, YTest, nb_cv):

    #-------------------------------------
    # Régression logistique - classification 
    #-------------------------------------
    #Temps de calcul
    start = time.time()
    
    #Instanciation de la régression logistique
    reglog=LogisticRegression(solver='saga',max_iter=6000)
    
    #Recherche des paramètre optimaux
    parametres = {'C':[0.5,1,2,3], 'penalty':['l1','l2']} #Paramètres à tester 
    reglog_opt=GridSearchCV(reglog, parametres)
     
    #Fit sur les données d'apprentissage
    reglog_opt.fit(XTrain,YTrain)
    
    #Meilleurs parametres
    bestparams = reglog_opt.best_params_
    
    #------------------------                  
    #   Validation croisée
    #----------------------- 
    scores = cross_val_score(reglog_opt, expli, cible, cv=nb_cv, scoring='accuracy')
    score_moyen=np.mean(scores)
    
    #----------------------------
    #   Prédiction
    #----------------------------
    y_pred=reglog_opt.predict(XTest)
    
    
    #-------------------------------
    #   Affichage des métriques
    #-------------------------------
    #Matrice de confusion
    matconf=metrics.confusion_matrix(YTest,y_pred)
    
    #Rappel et précision
    if len(np.unique(cible))>2:
        precision = precision_score(YTest,y_pred,average='weighted')
        rappel = recall_score(YTest,y_pred,average='weighted')
    else :
        precision = precision_score(YTest,y_pred,pos_label=np.unique(cible)[1])
        rappel = recall_score(YTest,y_pred,pos_label=np.unique(cible)[1])
    
    
    #-------------------------------
    #   Graphiques
    #-------------------------------
    #Courbe ROC 
    #Calcul de la probabilité d'appartenance à chaque classe
    proba_positif =  reglog_opt.predict_proba(XTest)
  
    if (len(np.unique(cible)) > 2):
        # Affichage courbe ROC
        y_onehot = pd.get_dummies(YTest, columns=reglog_opt.classes_)
    
        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1 )

        for i in range(proba_positif.shape[1]):
            y_true = y_onehot.iloc[:, i]
            y_score = proba_positif[:, i]

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)

            name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

        fig.update_layout(
            xaxis_title='Taux de faux positifs',
            yaxis_title='Taux de vrais positifs',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700, height=700)

    else : 
        fp, vp, _ = roc_curve(YTest, proba_positif[:,1],pos_label=np.unique(cible)[1])
        
        #calcul de l'aire sous la courbe
        aire_courbe = roc_auc_score(YTest, proba_positif[:,1])
        
        # Affichage courbe ROC
        fig = px.area(
            x=fp, y=vp,
            title=f'Courbe ROC modalité={np.unique(cible)[1]} (AUC={aire_courbe:.4f})',
            labels=dict(
                x='Taux de faux positifs', 
                y='Taux de vrais positifs'), width=700, height=700)
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1)

    # Temps de calcul
    end = time.time()
    temps = end - start
    
    #-------------------------------
    #    RETURN
    #-------------------------------
    return(score_moyen, matconf, temps, bestparams, precision, rappel, fig)
