import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

import utils

# create dash app
app = dash.Dash()
#server = app.server # uncomment this when pushing to heroku


# dataset options
datasets = ['Iris', 'Wine', 'Breast Cancer']
dataset_options = [dict(label=i, value=i) for i in datasets]

# model options
models = ['Logistic Regression', 'Random Forest', 'SVM']
model_options = [dict(label=i, value=i) for i in models]

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(id='dataset-header'),
                dcc.Dropdown(
                    id='dataset-dropdown',
                    options=dataset_options,
                    value='Iris'
                ),
                html.Button(
                    id='dataset-button',
                    children='Submit',
                    n_clicks=0
                )
            ],
            style={
                'width': '20%',
                'display': 'inline-block'
            }
        ),
        html.Div(
            [
                html.H1(id='model-header'),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=model_options,
                    value='Logistic Regression'
                ),
                html.Button(
                    id='model-button',
                    children='Submit',
                    n_clicks=0
                )
            ],
            style={
                'width': '20%'
            }
        ),
        html.Div(
            [
                html.P(id='model-acc'),
                html.P(id='model-f1')
            ]
        ),
        html.Div(
            [
                dash_table.DataTable(
                    id='table'
                )
            ],
            style={
                'width': '70%'
            }
        )
    ]
)


@app.callback(
    [
        Output('dataset-header', 'children'),
        Output('table', 'data'),
        Output('table', 'columns')
    ],
    [
        Input('dataset-button', 'n_clicks')
    ],
    [
        State('dataset-dropdown', 'value')
    ]
)
def update_dataset(n_clicks, dataset):
    data = utils.load_data(dataset)
    global x
    x = pd.DataFrame(data['data'], columns=data['feature_names'])
    global y
    y = data['target']

    h1 = 'Loaded Data: ' + dataset.capitalize()

    columns = [{'name': i, 'id': i} for i in x.columns]

    return h1, x.to_dict('records'), columns


@app.callback(
    [
        Output('model-header', 'children'),
        Output('model-acc', 'children'),
        Output('model-f1', 'children')
    ],
    [
        Input('model-button', 'n_clicks')
    ],
    [
        State('model-dropdown', 'value')
    ]
)
def update_model(n_clicks, model):
    h1 = 'Selected Model: ' + model
    model = utils.load_model(model)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
    model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    acc = model.score(x_test, y_test)
    f1 = f1_score(y_test, y_hat, average='micro')

    
    acc = f'Model Acc: {acc}'
    f1 = f'Model F1: {f1}'

    return h1, acc, f1


if __name__ == '__main__':
    app.run_server()
