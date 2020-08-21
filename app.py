import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import State, Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px
import time
import pandas as pd
import os
import networkx as nx

import requests
import pandas as pd
import colorlover as cl
import numpy as np
import json
from networkx.readwrite import json_graph
import igraph as ig
import uuid
from requests.packages.urllib3.util.retry import Retry

from LRUCache import LRUCache
from TimeoutHTTPAdapter import TimeoutHTTPAdapter
from controller.measures import measures, get_measures

http = requests.Session()

retries = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = TimeoutHTTPAdapter(max_retries=retries)
http.mount('http://', adapter)

http.mount('https://', adapter)

class Libraries:
    def __init__(self, api_file_path='./api.txt'):
        
        self.api_file_path = api_file_path        
        self.api_key = None
        self.payload = None
        self.url = None
        self.r  = None
        
        self.load_api_key()

        self.package_cache = {}
        self.response_caches = LRUCache(150)

        return
    
    def load_api_key(self):
        if 'API_KEY' in os.environ:
            self.api_key = os.environ['API_KEY']
        else:
            with open(self.api_file_path, 'r') as file:
                self.api_key = file.read()
        return
    
    def init_api_key(self, paramenters=None):
        self.payload = dict()
        self.payload.update({'api_key': self.api_key})
        return
    
    def get_response(self):
        self.r = self.get_response_for_payload(self.url)

    def get_response_for_payload(self, url):
        resp = self.response_caches.get(url)
        if resp is None:
            resp = get_response(url, self.payload)
            self.response_caches.put(url, resp)
        return resp

    def get_processed_package(self, package_name='requests'):
        if package_name not in self.package_cache:
            package = self.get_package(package_name)

            package['dependencies'] = self.get_dependencies(package)['dependencies']
            package['dependency_graph'] = self.preorder_label_parent(package, True)
            package['dependency_count'] = len(package['dependency_graph'][0])

            repo = self.get_repository(package)
            if repo is not None:
                package.update(repo)

            self.package_cache[package_name] = package
        return self.package_cache[package_name]

    def get_package(self, package_name='requests'):
        self.url = 'https://libraries.io/api/search?q={}'.format(package_name)
        self.get_response()
        results = self.r.json()
        filtered_results = list(filter(lambda obj: obj['name'].lower() == package_name.lower() and obj['stars'] > 0, results))
        json = filtered_results[0] if len(filtered_results) > 0 else results[0]
        del json['versions']
        del json['normalized_licenses']
        del json['keywords']
        del json['latest_stable_release']
        return json

    def get_dependencies(self, obj):
        self.url = 'https://libraries.io/api/{}/{}/latest/tree'.format(obj['platform'], obj['name'])
        self.get_response()
        json = self.r.json()
        return json

    def get_repository(self, obj):
        repo_path = obj['repository_url'].replace('https://', '').replace('.com', '')
        if 'github' in repo_path:
            self.url = 'https://libraries.io/api/{}'.format(repo_path)
            self.get_response()
            json = self.r.json()
        else:
            json = None
        return json

    def preorder_label_parent(self, parent, is_tree=False, node_list=None, links=None):
        if node_list is None:
            node_list=list()
        if links is None:
            links=[] 
        if 'id' not in parent:
            parent['id'] = parent['name'] if not is_tree else str(uuid.uuid4())
        if next((x for x in node_list if x['id'] == parent['id']), None) is None:
            node_list.append(parent)
        if 'dependencies' in parent:
            for child in parent.get('dependencies'):
                child['name'] = child['dependency']['project_name']
                child['id'] = child['name'] if not is_tree else str(uuid.uuid4())
                links.append((parent['id'], child['id']))
                self.preorder_label_parent(child, is_tree, node_list, links)
            
        return node_list, links

lib = Libraries()
lib.init_api_key()

def get_response(url, payload=None):
    counter = 1
    while counter < 10:
        try:
            r = requests.get(url, params=payload)
            r.raise_for_status()
            print(r.url)
            return r
        except Exception as e:
            print(e)
            counter = counter + 1
            time.sleep(100)
    raise Exception('No response from libraries.io inspite of multiple calls')

app = dash.Dash(
    __name__,
    meta_tags=[
        {
            'name': 'viewport',
            'content': 'width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no',
        }
    ],
)
server = app.server

app.config['suppress_callback_exceptions'] = True

# packages = ['vega', 'seaborn', 'plotly', 'd3', 'dash', 'bokeh', 'netflix-migrate', 'kafka-streams', 'ggplot', 'altair', 'matplotlib', 'pillow', 'jinja2', 'scipy', 'google-cloud-storage', 'redcarpet', 'django']
packages = ['dash', 'plotly.js', 'vega', 'seaborn', 'd3', 'matplotlib', 'ggplot', 'plotly', 'altair']

def get_palette(size):
    '''Get the suitable palette of a certain size'''
    if size == 2:
        palette = ['red', 'blue']
    elif size <= 8:
        palette = cl.scales[str(max(3, size))]['qual']['Set1']
    else:
        palette = cl.interp(cl.scales['8']['qual']['Set1'], size)
    return palette 

# Source: https://plotly.com/python/network-graphs/
def generate_parallel_coordinates(selected_packages, selected_measures):
    data_list = []
    for package_name in selected_packages:
        data = lib.get_processed_package(package_name=package_name)
        data_list.append(data)

    df = pd.DataFrame(data=data_list)
    dimensions = []
    for measure in selected_measures:
        measure_value = df[measure]
        measure_range = [max(measure_value), min(measure_value)] if measure == "dependency_count" else [min(measure_value), max(measure_value)]
        dimensions.append(dict(range =measure_range,
                               label = measure, values =measure_value, tickvals=np.unique(measure_value).tolist()))

    palette = get_palette(df.shape[0])
    parcoords = go.Parcoords(
        dimensions = list(dimensions),
        line = dict(color = df.index, 
        colorscale = palette),
        tickfont = dict(color ='#000', size=18),
        labelfont = dict(size=18)
    )

    data = [parcoords]

    layout = go.Layout(
        plot_bgcolor='#171b26',
        paper_bgcolor='#171b26',
        font_color='#FFF',
        font_size=18,
        font_family='"Open Sans", "HelveticaNeue", "Helvetica Neue", Helvetica, Arial, sans-serif',
        xaxis=dict(
            showticklabels=False
        ),
        yaxis=dict(
            showticklabels=False
        )
    )
    
    for i, package in enumerate(selected_packages):
        trace_dummy = go.Scatter(
            x=[0, 0, 0], # Data is irrelevant since it won't be shown
            y=[0, 0, 0],
            name=package,
            showlegend=True,
            marker=dict(color=palette[i], size=30)
        )
        data.append(trace_dummy)

    return {'data': data, 'layout': layout}

def get_plotly_data(E, coords):
    # E is the list of tuples representing the graph edges
    # coords is the list of node coordinates  assigned by a graph layout algorithm
    N = len(coords)
    Xnodes = [coords[k][0] for k in range(N)]# x-coordinates of nodes
    Ynodes = [coords[k][1] for k in range(N)]# y-coordnates of nodes

    Xedges = []
    Yedges = []
    for e in E:
        Xedges.extend([coords[e[0]][0], coords[e[1]][0], None])# x coordinates of the nodes defining the edge e
        Yedges.extend([coords[e[0]][1], coords[e[1]][1], None])# y - ' - 
        
    return Xnodes, Ynodes, Xedges, Yedges 

def get_node_trace(x, y, labels, marker_size=5, marker_color='#6959CD', line_color='rgb(50,50,50)', line_width=0.5):
    return go.Scatter(
                x=x,
                y=y,
                mode='markers',
                marker=dict(
                            size=marker_size, 
                            color=marker_color,
                            line=dict(color=line_color, width=line_width)
                             ),
            text=labels,
            hoverinfo='text'
               )

def get_edge_trace(x, y, linecolor='rgb(210,210,210)', linewidth=1):
    return go.Scatter(
                x=x,
                y=y,
                mode='lines',
                line=dict(color=linecolor, width=linewidth),
                hoverinfo='none'
               )

def get_package_tooltip(obj):
    tool_tip = obj['name']
    try:
        js = obj['version']
        name = "<B>Name:</B> {name}<BR>".format(name=obj['name']) if 'name' in obj else ""
        version = "<B>Version:</B> {version}<BR>".format(version=js['number']) if 'number' in js else ""
        lic = "<B>License:</B> {license}<BR>".format(license=js['original_license']) if 'original_license' in js else ""
        updated = "<B>Updated:</B> {updated}<BR>".format(updated=js['updated_at']) if 'updated_at' in js else ""
        tool_tip = name + version + lic + updated
    except Exception as e:
        print(e)
    return tool_tip

def generate_dependency_graph(selected_packages):
    package_data = []
    for package_name in selected_packages:
        package = lib.get_processed_package(package_name=package_name)
        package_data.append(package)

    graphs = []
    palette = get_palette(len(selected_packages))
    for i, package in enumerate(package_data):
        dendency_graph = package['dependency_graph']
        node_list, pre_links = dendency_graph[0], dendency_graph[1]

        g = ig.Graph(directed=True)
        g.add_vertices([o['id'] for o in node_list])
        g.add_edges(pre_links)
        ig_layout = g.layout('rt_circular') #(rt is Reingold-Tilford layout algorithm)
        E = [e.tuple for e in g.es]
        Xn, Yn, Xe, Ye =get_plotly_data(E, ig_layout)

        shapes=[]          
        for e in E:
            #set the coordinates of the Bezier control points b_i(xi, yi), for the cubic Bezier representing a tree edge
            x0 = Xn[e[0]] 
            y0 = Yn[e[0]]
            r0 = np.sqrt(x0**2+y0**2)
            x3 = Xn[e[1]] 
            y3 = Yn[e[1]] 
            r3 = np.sqrt(x3**2+y3**2)
            r = 0.5*(r0+r3)
            if r0 == 0 or r3 == 0:# i.e b0 or b3 is the tree root; in this case plot the edge as a segment of line
                #rm=max([r0, r3])
                x1 = x0
                y1 = y0
                x2 = x3
                y2 = 0.2*y0+y3*0.8
            else:
                x1 = r*x0/r0
                y1 = r*y0/r0
                x2 = r*x3/r3
                y2 = r*y3/r3
            
            shapes.append(dict(type='path',
                            layer='below',
                            #def SVG Bezier path representing an edge
                            path=f'M{x0} {y0}, C {x1} {y1}, {x2} {y2}, {x3} {y3}',
                            line=dict(color='rgb(210,210,210)', width=1)
                                ))                      
        title = dict(text='{}'.format(package['name']), font=dict(color='#FFF'))
        width = 500
        height = 500
        layout = go.Layout(title=title,
                        font= dict(size=16, family='"Open Sans", "HelveticaNeue", "Helvetica Neue", Helvetica, Arial, sans-serif'),
                        showlegend=False,
                        autosize=False,
                        width=width,
                        height=height,
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),          
                        hovermode='closest',
                        plot_bgcolor='#171b26',
                        paper_bgcolor='#171b26',
                        shapes=shapes)
        trace0 = go.Scatter(#type='scatter',
                x=[ig_layout[0][0]],
                y=[ig_layout[0][1]],
                marker=dict(size=10, color=palette[i]),
                text=[package_name],
                hoverinfo='text')
        #trace1 = get_edge_trace(Xe, Ye)
        trace2 = get_node_trace(Xn, Yn, marker_size=5, marker_color=palette[i], 
                            labels=[get_package_tooltip(o) for o in node_list])

        graph = dcc.Graph(
            className='dependencies-graph',
            figure={
                'data': [trace2, trace0],
                'layout': layout
            },
        )

        graphs.append(graph)
    return graphs


app.layout = html.Div(
    className='container scalable',
    children=[
        html.Div(
            id='banner',
            className='banner',
            children=[
                html.H6('Package Evaluation Dashboard'),
            ],
        ),
        html.Div(id='upper-container',
                 children=html.Div(
                    className='row',
                    children=[
                        html.Div(
                            className='six columns',
                            children=[
                                html.Label('Select packages', className='section-title', style={'color': '#FFF'}),
                                html.Div(
                                    id='package-select-dropdown-outer',
                                    style={'color': '#FFF'},
                                    children=dcc.Dropdown(
                                        id='package-select', multi=True, searchable=True, style={'color': '#FFF'},
                                        value=packages[:4], options=[{'label': i, 'value': i} for i in packages],
                                        persistence_type='local', clearable=True
                                    ),
                                )
                            ],
                        ),
                        html.Div(
                            className='six columns',
                            children=[
                                html.Label('Select measures', className='section-title'),
                                html.Div(
                                    id='measure-select-dropdown-outer',
                                    className="padding-bottom",
                                    children=dcc.Dropdown(
                                        id='measure-select', multi=True, searchable=True, style={'color': '#FFF'}, value=measures[:4], options=get_measures()
                                    ),
                                ),
                                html.Div(
                                    id='measure-checklist-container',
                                    children=dcc.Checklist(
                                        id='measure-select-all',
                                        options=[{'label': 'Select all', 'value': 'All'}],
                                        value=[],
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),
        )
        ,
        html.Div(
            id='lower-container',
            className='row',
            children=[
                html.Div(
                    id='parallel-coordinates-outer',
                    className='twelve columns',
                    children=[
                        html.P('Package comparison'),
                        html.Div(
                            id='parallel-coordinates-loading-outer',
                            children=[
                                dcc.Loading(
                                    id='loading',
                                    children=[dcc.Graph(
                                        id='parallel-coordinates',
                                        figure={
                                            'data': [],
                                            'layout': dict(
                                                plot_bgcolor='#171b26',
                                                paper_bgcolor='#171b26',
                                            ),
                                        },
                                    ),]
                                )
                            ],
                        ),
                        html.Div(
                            id='table-upper',
                            children=[
                                dcc.Loading(children=html.Div(id='packages-table-container', children=dash_table.DataTable(id='packages-table'))),
                            ],
                            className="padding-bottom"
                        ),
                        html.Div(
                            children=dcc.Loading(
                                children=[
                                html.P(
                                    className='chart-title',
                                    children='Dependency graphs',
                                ),
                                html.Div(
                                    id='dependency-graph-container',
                                    children=[]
                                )
                            ]),
                        )
                    ]
                )
            ]
        ),
        html.Div(
            id='footer',
            className='row',
            children=html.Div(
                className='twelve columns',
                children=
                    html.Div(
                        children=['Powered by ', dcc.Link('Libraries.io', href='https://libraries.io/')]
                    ),
            )
        )
    ],
)

@app.callback(
    dash.dependencies.Output('package-select', 'options'),
    [dash.dependencies.Input('package-select', 'search_value')],
    [dash.dependencies.State('package-select', 'value')],
)
def update_multi_options(search_value, value):
    all_values = None
    if not search_value:
        raise PreventUpdate
    try:
        if len(search_value) < 3:
            return [{'label': i, 'value': i} for i in packages]
        req_url = 'https://libraries.io/api/search?q={lib_name}&sort=stars&per_page=5'.format(lib_name=search_value)
        res = lib.get_response_for_payload(req_url).json()
        new_packages = [r['name'] for r in res]
        search_url = 'https://libraries.io/api/search?q={}&per_page=3'.format(search_value)
        exact_search = lib.get_response_for_payload(search_url).json()
        if search_value in [r['name'] for r in exact_search]:
            new_packages.append(search_value)
        all_packages = list(set(value + new_packages))
        all_packages.sort(key=len)
        all_values = [{'label': r, 'value': r} for r in all_packages]

    except Exception as e:
        print(e)
    return all_values or [{'label': i, 'value': i} for i in packages]

@app.callback(
    Output('measure-select', 'value'),
    [Input('measure-select', 'options'), Input('measure-select-all', 'value')],
)
def update_measure_dropdown(options, select_all):
    if select_all == ['All']:
        value = [i['value'] for i in options]
    else:
        value = dash.no_update
    return value

# Updates the Select All checkbox as the measure select is updated
@app.callback(
    Output('measure-checklist-container', 'children'),
    [Input('measure-select', 'value')],
    [State('measure-select', 'options'), State('measure-select-all', 'value')],
)
def update_measure_checklist(selected, select_options, checked):
    if len(selected) < len(select_options) and len(checked) == 0:
        raise PreventUpdate()

    elif len(selected) < len(select_options) and len(checked) == 1:
        return dcc.Checklist(
            id='measure-select-all',
            options=[{'label': 'Select All', 'value': 'All'}],
            value=[],
        )

    elif len(selected) == len(select_options) and len(checked) == 1:
        raise PreventUpdate()

    return dcc.Checklist(
        id='measure-select-all',
        options=[{'label': 'Select All', 'value': 'All'}],
        value=['All'],
    )

@app.callback(
    Output('packages-table-container', 'children'),
    [
        Input('package-select', 'value'),
    ],
)
def update_packages_table(selected_packages):
    columns = [
        'name',
        'language',
        'platform',
        'licenses',
        'latest_release_published_at',
        'latest_release_number',
        'description'
    ]

    data_list = get_packages_data(selected_packages)

    data_list_reduced_columns =  pd.DataFrame(data_list, columns=columns)
    return dash_table.DataTable(
        id='packages-table',
        columns=[{'name': i, 'id': i} for i in columns],
        data=data_list_reduced_columns.to_dict('rows'),
        #filter_action='native',
        page_size=10,
        style_cell={'background-color': '#242a3b', 'color': '#FFF', 'text-align': 'left'},
        style_as_list_view=False,
        style_header={'background-color': '#1f2536', 'padding': '0px 5px'}
    )

def get_packages_data(selected_packages):
    data_list = []
    for package_name in selected_packages:
        data = lib.get_processed_package(package_name=package_name)
        data_list.append(data)
    return data_list

@app.callback(
    Output('parallel-coordinates', 'figure'),
    [
        Input('package-select', 'value'),
        Input('measure-select', 'value'),
    ],
)
def update_parallel_coordinates(selected_packages, selected_measures):
    return generate_parallel_coordinates(selected_packages, selected_measures)


@app.callback(
    Output('dependency-graph-container', 'children'),
    [
        Input('package-select', 'value'),
    ],
)
def update_dependency_graph(selected_packages):
    return generate_dependency_graph(
        selected_packages
    )

if __name__ == '__main__':
    app.run_server(debug=True)