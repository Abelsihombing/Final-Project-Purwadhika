import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import seaborn as sns
from dash.dependencies import Input, Output, State
import pickle
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


df = pd.read_csv('clean_data_solar1.csv')
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

loadModel = pickle.load(open('solar_xgb_model.sav', 'rb'))
loadTransform = pickle.load(open('solar_xgb_model.sav', 'rb'))

app.layout = html.Div(
    children=[
        html.H1('Dash Project Dashboard'),
        html.Div(children='''
        Created by: Abel Y.T Sihombing
    '''),
        dcc.Tabs(children=[
            dcc.Tab(value='Tab1', label='Machine Learning', children=[
                html.Div([
                    html.Div([
                        html.Div(children=[
                            html.P('Temperature (F): '),
                            dcc.Input(id='my-id-temp', value = '0', type = 'number')
                        ], className='col-4'),

                        html.Div(children=[
                            html.P('Pressure (decimal 2 angka di belakang koma): '),
                            dcc.Input(id='my-id-pres', value = '0', type = 'number')
                        ], className='col-4'),

                        html.Div([
                            html.P('Humidity (%): '),
                            dcc.Input(id='my-id-humi', value = '0', type = 'number')
                        ], className='col-4')
                    ], className='row'),

                    html.Br(),
                    html.Div([
                        html.Div(children=[
                            html.P('Wind Direction (Degrees(0-360)(2 angka di belakang koma)): '),
                            dcc.Input(id='my-id-wind', value = '0', type = 'number')
                        ], className='col-4'),

                        html.Div(children=[
                            html.P('Speed (2 angka dibelakang koma): '),
                            dcc.Input(id='my-id-spee', value = '0', type = 'number')
                        ], className='col-4'),

                        html.Div([
                            html.P('Total_TimeSun (11hr/12hr): '),
                            dcc.Input(id='my-id-tota', value = '11', type = 'number')
                        ], className='col-4')
                    ], className='row'),   

                    html.Br(),
                    html.Div([
                        html.Div(children=[
                            html.P('Month (9-12): '),
                            dcc.Input(id='my-id-mont', value = '9', type = 'number')
                        ], className='col-4'),

                        html.Div(children=[
                            html.P('Days (1-31): '),
                            dcc.Input(id='my-id-days', value = '1', type = 'number')
                        ], className='col-4'),

                        html.Div([
                            html.P('Hour (0-23): '),
                            dcc.Input(id='my-id-hour', value = '0', type = 'number')
                        ], className='col-4')
                    ], className='row')
                ]),
                
                html.Br(),
                html.Div(id = 'my-div')
            ]),#dcc.Tab

            dcc.Tab(value='Tab2', label='EDA', children=[
                html.Div([
                    html.Div(children = dcc.Graph(
                    id = 'pie chart',
                    figure = {
                        'data':[
                    go.Pie(labels = [i for i in df['direct'].unique()], 
                    values= [df[df['direct'] == i]['Radiation'].mean() for i in df['direct'].unique()]
                    )],
                    'layout': go.Layout(title = 'The Effect of Wind Direction on Radiation')}
                   
                            
                    ), className='col-6'),
                    html.Div(children = 
                    dcc.Graph(
                        id='graph',
                        figure={
                            'data': [
                                {'x': df['hour'], 'y': df['Temperature'], 'type': 'bar'}
                                
                            ],
                            'layout': go.Layout(
                                    xaxis={'title':'Hour'},
                                    yaxis={'title':'Temperature'},
                                    title='The Effect of Hour on Temperature',
                                    hovermode='closest'
                            )}
                    ), className='col-6')
                ], className='row'),
                html.Div([
                    html.Div(children = dcc.Graph(
                            id = 'graph-scatter',
                            figure = {'data':[
                                go.Scatter(
                                    x = df['Temperature'],
                                    y = df['Radiation'],
                                    mode='markers')        
                                ],
                                'layout':go.Layout(
                                    xaxis={'title':'Temperature'},
                                    yaxis={'title':'Radiation'},
                                    title='The Effect of Temperature on Radiation',
                                    hovermode='closest'
                                )}
                    ),className='col-8'),
                    html.Div(children = 
                    dcc.Graph(
                        id='example',
                        figure={
                            'data': [
                                {'x': df['Month_cat'], 'y': df['Radiation'], 'type': 'bar'},
                                
                            ],
                            'layout': go.Layout(
                                    xaxis={'title':'Hour'},
                                    yaxis={'title':'Radiation'},
                                    title='The Effect of Hour on Radiation',
                                    hovermode='closest'
                            )}
                    ),className='col-4'),
                ],className='row'),

                html.Div(children = 
                    dcc.Graph(
                        id='example-graph',
                        figure={
                            'data': [
                                {'x': df['hour'], 'y': df['Radiation'], 'type': 'bar'},
                                
                            ],
                            'layout': go.Layout(
                                    xaxis={'title':'Hour'},
                                    yaxis={'title':'Radiation'},
                                    title='The Effect of Hour on Radiation',
                                    hovermode='closest'
                            )}
                    ))
            ])#dcc.Tab
        ], content_style= {
            'fontFamily': 'Arial',
            'borderBottom': '1px solid #d6d6d6',
            'borderLeft': '1px solid #d6d6d6',
            'borderRight': '1px solid #d6d6d6',
            'padding': '44px'
        }
        )#dcc.Tabs

])

@app.callback(
    Output('my-div', 'children'),
    [Input('my-id-temp', 'value'),
     Input('my-id-pres', 'value'),
     Input('my-id-humi', 'value'),
     Input('my-id-wind', 'value'),
     Input('my-id-spee', 'value'),
     Input('my-id-tota', 'value'),
     Input('my-id-mont', 'value'),
     Input('my-id-days', 'value'),
     Input('my-id-hour', 'value')
     ]
    )


def update_output_div(my_id_temp, my_id_pres, my_id_humi, my_id_wind, 
    my_id_spee, my_id_tota, my_id_mont, my_id_days, my_id_hour):
    data_baru = pd.DataFrame(data = [(float(my_id_temp), float(my_id_pres), 
        float(my_id_humi), float(my_id_wind), float(my_id_spee), 
        int(my_id_tota), int(my_id_mont), int(my_id_days), int(my_id_hour))], 
        columns = ['Temperature', 'Pressure', 'Humidity', 
        'WindDirection(Degrees)', 
        'Speed', 'Total_TimeSun', 'Month', 'Days', 'hour'])

    # loadTransform = pickle.load(open('solar_xgb_model.sav', 'rb'))
    polynomial_features = PolynomialFeatures(degree=2)
    x_pol = polynomial_features.fit_transform(data_baru)
    predict = loadModel.predict(x_pol)
    return '\n\nHasil Prediksi adalah: {}'.format(predict)

if __name__ == '__main__':
    app.run_server(debug=True)