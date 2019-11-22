# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:03:15 2019

@author: Aditya Ganesh
"""


# import necessary libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd #Provides R like data structures and a high level API to work with data
import warnings # Ignore  the warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import sklearn.linear_model as sk
from sklearn import metrics

def read_data(data):
    X = data.loc[:, data.columns != 'Status']
    y = data.loc[:, 'Status']
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=42)
    return Xtrain, Xtest, ytrain, ytest


# get results after logistic regression
def get_results(data, C=0.1, max_iter=100, penalty='l1'):
    Xtrain, Xtest, ytrain, ytest = read_data(data)
    model = sk.LogisticRegression(C = C, max_iter=max_iter, penalty=penalty)
    model.fit(Xtrain,ytrain) 
    accuracy = model.score(Xtest, ytest)
    auc_val = metrics.roc_auc_score(ytest, model.predict_proba(Xtest)[:,1])
    print(accuracy , " : ", auc_val)
    cnf = metrics.confusion_matrix(ytest, model.predict(Xtest))
    print(cnf)
    return accuracy, auc_val, cnf, model, Xtest, ytest


def serve_roc_curve(model,X_test,y_test):
    probs = model.predict_proba(X_test)[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, probs)

    # AUC Score
    auc_score = metrics.roc_auc_score(y_true=y_test, y_score=probs)

    trace0 = go.Scatter(
        x=fpr, 
        y=tpr,
        mode='lines',
        name='Test Data',
    )

    layout = go.Layout(
        title=f'ROC Curve (AUC = {auc_score:.3f})',
        xaxis=dict(
            title='False Positive Rate'
        ),
        yaxis=dict(
            title='True Positive Rate'
        ),
        legend=dict(x=0, y=1.05, orientation="h"),
        margin=dict(l=50, r=10, t=55, b=40),
    )

    data = [trace0]
    layout = go.Layout(
                title=f'ROC Curve (AUC = {auc_score:.3f})',
                xaxis=dict(
                    title='False Positive Rate'
                ),
                yaxis=dict(
                    title='True Positive Rate'
                ),
                legend=dict(x=0, y=1.05, orientation="h"),
                margin=dict(l=50, r=10, t=55, b=40),
            )

    data = [trace0]
    figure = go.Figure(data=data, layout=layout)

    return figure


# read data into file and impute missing values
data= pd.read_csv("new_outbreak.csv")
accuracy, auc, cnf, model, Xtest, ytest = get_results(data,C = 0.1, max_iter = 100, penalty='l1')
#roc_figure = serve_roc_curve(model=model,X_test=Xtest,y_test=ytest)


my_css = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=my_css)
server = app.server
app.layout = html.Div(children=[html.Label('C: '),
            dcc.Dropdown(
        id='C',
        options = [
                {'label': 0.001, 'value': 0.001},
                {'label': 0.1, 'value': 0.1},
                {'label': 1, 'value': 1},
                {'label': 10, 'value': 10},
                {'label': 100, 'value': 100},
                {'label': 1000, 'value': 1000}
                ],
        value=0.001
    ),html.Label('MAX_ITER: '),
    dcc.Dropdown(
        id='max_iter',
        options = [
                
        {'label': 100, 'value': 100},
        {'label': 110, 'value': 110},
        {'label': 120, 'value': 120},
        {'label': 130, 'value': 130},
        {'label': 140, 'value': 140}
        ],
        value=100
    ),html.Label('PENALTY: '),
    dcc.Dropdown(
        id='penalty',
        options = [
        {'label': 'l1', 'value': 'l1'},
        {'label': 'l2', 'value': 'l2'},
        ],
        value='l1'
        ), html.H1(children='MODEL METRICS'),
              html.Div([html.Div(id='model'),
                        dcc.Graph(id='graph-line-roc-curve')]), 
                
])
    
@app.callback(
         Output('graph-line-roc-curve','figure'),
    [Input('C', 'value'),Input('max_iter','value'),Input('penalty','value')])
def update_graph(x,y,z):
    accuracy, auc_val, cnf, model, Xtest, ytest = get_results(data, C = x, max_iter = y, penalty=z)
    roc_figure = serve_roc_curve(model=model,X_test=Xtest,y_test=ytest)
    return roc_figure

 
@app.callback(
    Output('model','children'),
    [Input('C', 'value'),Input('max_iter','value'),Input('penalty','value')])
def update_table(x,y,z):
    accuracy, auc_val, cnf, model, Xtest, ytest = get_results(data,C = x, max_iter = y, penalty=z)
    st=[]
    st.append(html.P("ACCURACY = " + str(accuracy)))
    st.append(html.P("AUC = "+ str(auc_val)))
    st.append(html.P("CONFUSION MATRIX VALUES"))
    st.append(html.P("TP = " + str(cnf[0][0])))
    st.append(html.P("FP = " + str(cnf[0][1])))
    st.append(html.P("FN = " + str(cnf[1][0])))
    st.append(html.P("TN = " + str(cnf[1][1])))
    st.append(html.P("___")) 
    return st
 
if __name__ == '__main__':
    app.run_server(debug=True)
