#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 17:13:29 2021

@author: learegazzetti
"""

############################################ Import des packages ########################################
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import base64
import io
import pandas as pd
import dash_table
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import Classification
import Regression



################################################# Interface #############################################

# Initialisation de Dash
app = dash.Dash(__name__)

# Page d'accueil initiale
app.layout = html.Div(style={'backgroundColor': '#FFFFFF'}, children=[
    dcc.Location(id='url', refresh=False),
    html.H1(
        children='Bienvenue',
        style={
            'textAlign': 'center',
            'color': '#0000FF'
        }
    ),
    html.Div(children='Cette interface vous permet d\'appliquer des algorithmes de machine learning sur le jeu de données de votre choix.'  , 
             style={
        'textAlign': 'center',
        'color': '#0000FF',
        'font-size' : '20px'
    }),
    
    html.Br(),
    
    dcc.Upload(
            id="upload-data",
            children=html.Div(
                ["Cliquez ici pour sélectionner votre fichier."]
            ),
            style={
                "width": "98%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px"
            },
            multiple=True
        ),
        html.Div(id="output")
        , html.Br(),
        html.Div(id="output_var"),
        html.Br(),
        html.Div(id="output_submit"),
        html.Br(),
        html.Div(id="output_choix"),
        html.Br(),
        dcc.Graph(id="graph"),
        #html.Br(),
        html.Div(id="output_algo")
    
])


# Affichage du fichier de données
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    
    decoded = base64.b64decode(content_string)

    if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=',')
            
            return html.Div(id="load_data", children =[html.H4(children = "Voici un aperçu de " + str(filename) + " : ",
                                                               style={
        'textAlign': 'left',
        'color': 'black'
        }),
                               
        # Affichage d'un extrait du jeu de données                       
        dash_table.DataTable(id='table',
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            page_size=10),
        
        # Choix de la variable cible
        html.Br(),
        html.Label('Sélectionner la variable cible : '),
        dcc.Dropdown(id="cible",
            options=[{'label': i, 'value': i} for i in df.columns]),
    
        # Choix des variables explicatives  
        html.Br(),
        html.Label('Sélectionner la(les) variable(s) explicative(s) : '),
        dcc.Dropdown(id="explicatives",options=[{'label': i, 'value': i} for i in df.columns], multi=True),
        html.Br()
        ])
    
    else :
        return(html.Div(children=[html.H2(
            children='Attention : le fichier de données doit être au format csv avec une virgule pour séparateur.',
            style={
                'textAlign':'center',
                'color':'darkred'
                })])
            
        )   
    
    
# Mise à jour de l'affichage lors d'un nouveau fichier de données
@app.callback(Output('output', 'children'),
              Input('upload-data', 'contents'),
              Input('upload-data', 'filename'))

def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n) for c, n in
            zip(list_of_contents, list_of_names)]
        return children
 

# Sélection de la variable cible et des variables explicatives
@app.callback(Output('output_var', 'children'),
              Input('cible', 'value'),
              Input('explicatives','value'))
    
def set_var(selected_cible,selected_explicatives):
    return (html.Div('Variable cible sélectionnée : {}'.format(selected_cible), style={
        'textAlign': 'center',
        'color': '#57080F',
        'font-size' : '18px'}), 
            html.Br(), 
            html.Div('Variables explicatives sélectionnées : {}'.format(selected_explicatives),
                    style={
        'textAlign': 'center',
        'color': '#57080F',
        'font-size' : '18px'}),
            html.Br(),
        html.Div('Une fois les variables sélectionnées, cliquez sur le bouton ci-dessous :', style={
        'textAlign': 'left',
        'color': 'black',
        'font-size' : '20px'}), 
        html.Button('Envoyer', id='submit_button', n_clicks=0, style={
                "textAlign": "center",
                "margin": "10px",
                "font-size":"16px",
                "margin-left": "550px",
                
            }))


# Choix de l'algorithme en fonction du type (quali ou quanti) de la variable cible
@app.callback(Output('output_submit', 'children'),
              Input('submit_button','n_clicks'),
              Input('table', 'data'),
              State('cible', 'value'))

def submit(n_clicks, d, value):
    if n_clicks >= 1:
        x = pd.DataFrame(d)
        
        # Algorithmes de classification
        if (x[value].dtypes) == object:
            return (html.Div("Vous pouvez sélectionner ici le nombre de folds pour la validation croisée (par défaut 10)"),
                    dcc.Slider(id="slider_folds", min = 2, max = x.shape[0]*0.7, step = 1, value = 10, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Br(),
                    html.Div("La variable cible est qualitative. Vous pouvez choisir parmi les algorithmes suivants : "), 
                    html.Br(),
                    dcc.RadioItems(id="choix_algo", options = [{'label':'Arbre de décision', 'value':'arbre'},
                                                                   {'label':'Analyse discriminate','value':'lda'},
                                                                   {'label':'Régression logistique','value':'reg_log'}],
                                   labelStyle={'display':'inline-block'}))
    
        # Algorithmes de regression
        elif ((x[value].dtypes) == int) or ((x[value].dtypes) == float):
            return (html.Div("Vous pouvez sélectionner ici le nombre de folds pour la validation croisée (par défaut 10)"),
                    dcc.Slider(id="slider_folds", min = 2, max = x.shape[0]*0.7, step = 1, value = 10, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Br(),
                    html.Div("La variable cible est quantitative. Vous pouvez choisir parmi les algorithmes suivants :"),
                    html.Br(),
                    dcc.RadioItems(id="choix_algo", options = [{'label':'Régression linéaire', 'value':'reg_lin'},
                                                                   {'label':'K plus proches voisins','value':'Knn'},
                                                                   {'label':'Arbre de régression','value':'DTReg'}],
                                   labelStyle={'display':'inline-block'}))
    else :
        return html.Div("Veuillez cliquer sur le bouton envoyer")          



# Resultats
@app.callback(Output('output_algo', 'children'),
              Output('graph','figure'),
              Input('choix_algo', 'value'),
              State('table','data'),
              State('cible','value'),
              State('explicatives','value'),
              State('slider_folds','value'))
            
def final(choix_algo, dd, Y, X, nb_folds):
    
    dd = pd.DataFrame(dd)
    Y = dd[Y]
    X = dd[X]
    
    
    #---------------------------------------
    # Recodage des variables qualitatives
    #---------------------------------------
    if sum(X.dtypes == np.object_) > 0 :
        #Recodage des variables qualitatives en disjonctif 0/1
        liste_quali=[var for var in X if X[var].dtype==np.object_]

        #Recodage des variables qualitatives
        df_Recodage= pd.get_dummies(X[liste_quali], drop_first=True)
    
        #Liste des variables quantitatives
        liste_quanti=[var for var in X if X[var].dtype!=np.object_]

        #Réunir les quantitatives et les variables qualitatives recodées
        X = pd.concat([X[liste_quanti],df_Recodage],axis=1)
        
    #---------------------------------------
    # Scindage en apprentissage et test
    #---------------------------------------
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)

    #---------------------------------------
    # Centrage - réduction des variables X
    #---------------------------------------
    #instanciation
    sc = StandardScaler()
        
    #transformation – centrage-réduction
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    
    # Affichage des métriques d'évaluation, temps de calcul et graphiques selon l'algorithme de classification
    if choix_algo == 'arbre' or choix_algo == 'lda' or choix_algo == 'reg_log':
        if choix_algo == 'arbre':
            res = Classification.Arbre_Decision(Y, X, X_train, X_test, y_train, y_test, nb_folds)
        
        if choix_algo == 'lda':
            res = Classification.Analyse_Discriminante(Y, X, X_train, X_test, y_train, y_test, nb_folds)
             
        if choix_algo == 'reg_log':
            res = Classification.Regression_Logistique(Y, X, X_train, X_test, y_train, y_test, nb_folds)
         
        conf_mat = res[1]
        conf_mat_df = pd.DataFrame(conf_mat, columns = [str(i) for i in range(conf_mat.shape[1])])

        return html.Div(children = [
            html.Div("Nombre de folds sélectionnés : {}".format(nb_folds)),
            html.Br(),
            html.Div("Temps de calcul (en secondes) : {}".format(round(res[2], 1))),
            html.Br(),   
            html.Div("Meilleurs paramètres estimés en fonction de l'accuracy : {}".format(res[3])),
            html.Br(),
            html.Div("Pourcentage de bonnes prédictions en cross-validation (accuracy) : {}".format(round(res[0], 3))),
            html.Br(),
            html.Div("Precision : {}".format(round(res[4], 3))),
            html.Br(),
            html.Div("Rappel : {}".format(round(res[5], 3))),
            html.Br(),
            html.Div("Matrice de confusion : "),
                    dash_table.DataTable(id='mat_conf',
                    data=conf_mat_df.to_dict('records'), columns = [{'name': col, 'id': col} for col in conf_mat_df.columns]),
            
            ]) ,res[6]


    # Affichage des métriques d'évaluation, temps de calcul et graphiques selon l'algorithme de regression
    elif choix_algo == 'reg_lin' or choix_algo == 'Knn' or choix_algo == 'DTReg' :
    
        if choix_algo == 'reg_lin':
            res = Regression.Regression_Lineaire(Y, X, X_train, X_test, y_train, y_test, nb_folds)
    
        if choix_algo == 'Knn':
            res = Regression.K_Voisins(Y, X, X_train, X_test, y_train, y_test, nb_folds)
        
    
        if choix_algo == 'DTReg':
            res = Regression.Arbre_Regression(Y, X, X_train, X_test, y_train, y_test, nb_folds)

        
        return html.Div(children = [
            html.Div("Nombre de folds sélectionnés : {}".format(nb_folds)),
            html.Br(),
            html.Div("Temps de calcul (en secondes) : {}".format(round(res[0], 1))),
            html.Br(),  
            html.Div("Meilleurs paramètres estimés en fonction du r2 : {}".format(res[3])),
            html.Br(),
            html.Div("Coefficient de détermination en cross-validation : {}".format(round(res[1], 3))),
            html.Br(),
            html.Div("Erreur quadratique moyenne : {}".format(round(res[2], 3))),
            ]), res[4]


if __name__ == '__main__':
    app.run_server(debug=True)
  

