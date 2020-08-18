import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import State, Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px

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
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from TimeoutHTTPAdapter import TimeoutHTTPAdapter

http = requests.Session()

retries = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = TimeoutHTTPAdapter(max_retries=retries)
http.mount("http://", adapter)

http.mount("https://", adapter)


class Libraries:
    def __init__(self, api_file_path='./api.txt'):
        
        self.api_file_path = api_file_path        
        self.api_key = None
        self.payload = None
        self.url = None
        self.r  = None
        self.json = None
        self.json_flat = None
        
        self.load_api_key()

        self.dependency_cache = {}
        self.library_cache = {}
        
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
        self.r = get_response(self.url, self.payload)

    def get_package(self, package='requests'):
        if package not in self.library_cache:
            self.url = 'https://libraries.io/api/search?q={}'.format(package)
            self.get_response()
            results = self.r.json()
            # results.sort(key=lambda x: x.get('stars'), reverse=True)
            filtered_results = list(filter(lambda obj: obj['name'] == package, results))
            self.json = filtered_results[0] if len(filtered_results) > 0 else results[0]
            del self.json['versions']
            del self.json['normalized_licenses']
            del self.json['keywords']
            del self.json['latest_stable_release']
            self.library_cache[package] = self.json
        return self.library_cache[package]

    def get_dependencies(self, obj):
        if obj['name'] not in self.dependency_cache:
            self.url = 'https://libraries.io/api/{}/{}/latest/tree'.format(obj['platform'], obj['name'])
            self.get_response()
            self.json = self.r.json()
            self.dependency_cache[obj['name']] = self.json
            # print(dependencies)
        return self.dependency_cache[obj['name']]

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
    raise Exception("No response from libraries.io inspite of multiple calls")

app = dash.Dash(
    __name__,
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no",
        }
    ],
)
server = app.server

app.config["suppress_callback_exceptions"] = True

# libraries = ['netflix-migrate', 'kafka-streams', 'vega']
libraries = ['netflix-migrate', 'kafka-streams', 'vega', 'seaborn', 'bokeh', 'dash', 'plotly', 'ggplot', 'altair', 'matplotlib', 'pillow', 'jinja2', 'scipy', 'google-cloud-storage', 'redcarpet', 'django']
measures = ['rank', 'stars', 'dependents_count', 'dependent_repos_count', 'forks']

library_data = []
for package in libraries:
    data = lib.get_package(package=package)
    # print(data)
    library_data.append(data)

def build_upper_left_panel():
    return html.Div(
        id="upper-left",
        className="four columns",
        children=[
            html.Div(
                id="library-select-outer",
                children=[
                    html.Label("Select libraries"),
                    html.Div(
                        id="library-select-dropdown-outer",
                        children=dcc.Dropdown(
                            id="library-select", multi=True, searchable=True, style={'color': '#FFF'},
                            value=libraries[:4], options=[{"label": i, "value": i} for i in libraries]
                        ),
                    ),
                    html.Label("Select measures"),
                    html.Div(
                        id="measure-checklist-container",
                        children=dcc.Checklist(
                            id="measure-select-all",
                            options=[{"label": "Select All Measures", "value": "All"}],
                            value=[],
                        ),
                    ),
                    html.Div(
                        id="measure-select-dropdown-outer",
                        children=dcc.Dropdown(
                            id="measure-select", multi=True, searchable=True, style={'color': '#FFF'}, value=measures[:4], options=[{"label": i, "value": i} for i in measures]
                        ),
                    ),
                    html.Div(
                        id="table-upper",
                        children=[
                            html.P("Selected library details"),
                            dcc.Loading(children=html.Div(id="libraries-table-container", children=dash_table.DataTable(id='libraries-table'))),
                        ],
                    )
                ],
            ),
        ],
    )



def get_palette(size):
    '''Get the suitable palette of a certain size'''
    if size == 2:
        palette = ['red', 'blue']
    elif size <= 8:
        palette = cl.scales[str(max(3, size))]['qual']['Set2']
    else:
        palette = cl.interp(cl.scales['8']['qual']['Set2'], size)
    return palette 

# Source: https://plotly.com/python/network-graphs/
def generate_parallel_coordinates(library_select, measure_select):
    data_list = []
    for package in library_select:
        data = lib.get_package(package=package)
        data_list.append(data)

    df = pd.DataFrame(data=data_list)
    dimensions = []
    for measure in measure_select:
        dimensions.append(dict(range = [min(df[measure]),max(df[measure])],
                label = measure, values = df[measure], tickvals=np.unique(df[measure]).tolist()))
    data = go.Parcoords(
        dimensions = list(dimensions),
        line = dict(color = df.index, 
        colorscale = get_palette(df.shape[0]), 
        colorbar = dict(tickvals = np.unique(df.index).tolist(), 
        ticktext = np.unique(df['name']).tolist(), 
        nticks = df.shape[0],
        tickfont = dict(color ='#FFF', size=18))),
        tickfont = dict(size=18),
        labelfont = dict(color ='#FFF', size=18)
    )

    layout = go.Layout(
        plot_bgcolor="#171b26",
        paper_bgcolor="#171b26",
    )

    return {"data": [data], "layout": layout}

def preorder_label_parent(tree_dict, labels=None, links=None):
    if labels is None:
        labels=list()
    if links is None:
        links=[]
    parent_name = tree_dict['name']
    if parent_name not in labels:
        labels.append(parent_name)
    if 'dependencies' in tree_dict:
        for child in tree_dict.get('dependencies'):
            child['name'] = child['dependency']['project_name']
            links.append((parent_name, child['name']))
            preorder_label_parent(child, labels, links)

    return labels, links

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
        Yedges.extend([coords[e[0]][1], coords[e[1]][1], None])# y - " - 
        
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


tree_cache = {}


def generate_dependency_graph(selected_libraries):
    for package in selected_libraries:
        data = lib.get_package(package=package)
        library_data.append(data)

    graphs = []
    for package_name in selected_libraries:
        package = next(x for x in library_data if x['name'] == package_name)
        package_with_dependencies = lib.get_dependencies(package)
        package_with_dependencies['name'] = package_name
        # procedure_data = raw_data[
        #     raw_data["Hospital Referral Region (HRR) Description"].isin(library_select)
        # ].reset_index()

        # traces = []
        # selected_index = procedure_data[
        #     procedure_data["Provider Name"].isin(provider_select)
        # ].index

        # text = (
        #     procedure_data["Provider Name"]
        #     + "<br>"
        #     + "<b>"
        #     + procedure_data["DRG Definition"].map(str)
        #     + "/<b> <br>"
        #     + "Average Procedure Cost: $ "
        #     + procedure_data[cost_select].map(str)
        # )

        # hoverinfo="text",
        # hovertext=text,
        # selectedpoints=selected_index,
        # hoveron="points",

        # Create random graph
        #G = nx.random_geometric_graph(200, 0.125)
        # print(dependencies)
        if package_name not in tree_cache:
            tree_cache[package_name] = preorder_label_parent(package_with_dependencies)

        tree_graph = tree_cache[package_name]
        # print(tree_graph)
        pre_labels, pre_links = tree_graph[0], tree_graph[1]
        # print(pre_labels)
        # print(pre_links)
        # G = nx.DiGraph()
        # G = nx.random_geometric_graph(200, 0.125)
        # print(pre_labels)
        g = ig.Graph(directed=True)
        g.add_vertices(pre_labels)
        g.add_edges(pre_links)
        #g = ig.Graph.Tree(400, 5)
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
        title = dict(text="{}".format(package['name']), font=dict(color="#FFF"))
        width = 500
        height = 500
        layout = go.Layout(title=title,
                        font= dict(size=12),
                        showlegend=False,
                        autosize=False,
                        width=width,
                        height=height,
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),          
                        hovermode='closest',
                        plot_bgcolor="#171b26",
                        paper_bgcolor="#171b26",
                        shapes=shapes)
        trace0 = go.Scatter(#type='scatter',
                x=[ig_layout[0][0]],
                y=[ig_layout[0][1]],
                marker=dict(size=10, color='rgb(255,200,0)'),
                text=[package_name],
                hoverinfo='text')
        #trace1 = get_edge_trace(Xe, Ye)
        trace2 = get_node_trace(Xn, Yn, marker_size=5, marker_color='rgb(255,200,0)', 
                            labels=pre_labels)

        graph = dcc.Graph(
            className="dependencies-graph",
            figure={
                "data": [trace2, trace0],
                "layout": layout
            },
        )

        graphs.append(graph)
    return graphs


app.layout = html.Div(
    className="container scalable",
    children=[
        html.Div(
            id="banner",
            className="banner",
            children=[
                html.H6("Package Evaluation Dashboard"),
            ],
        ),
        html.Div(
            id="upper-container",
            className="row",
            children=[
                build_upper_left_panel(),
                html.Div(
                    id="parallel-coordinates-outer",
                    className="eight columns",
                    children=[
                        html.P(
                            id="map-title",
                            children="Comparison of libraries",
                        ),
                        html.Div(
                            id="parallel-coordinates-loading-outer",
                            children=[
                                dcc.Loading(
                                    id="loading",
                                    children=[dcc.Graph(
                                        id="parallel-coordinates",
                                        figure={
                                            "data": [],
                                            "layout": dict(
                                                plot_bgcolor="#171b26",
                                                paper_bgcolor="#171b26",
                                            ),
                                        },
                                    ),]
                                )
                            ],
                        ),
                    ],
                ),
            ],

        ),
        html.Div(
            id="lower-container",
            className="row",
            children=html.Div(
                className="twelve columns",
                children=[
                    html.P(
                        children="Dependency graphs",
                    ),
                    html.Div(
                        id="dependency-graph-container",
                        children=[]
                    )
                ],
            )
        )
    ],
)


@app.callback(
    dash.dependencies.Output("library-select", "options"),
    [dash.dependencies.Input("library-select", "search_value")],
    [dash.dependencies.State("library-select", "value")],
)
def update_multi_options(search_value, value):
    all_values = None
    if not search_value:
        raise PreventUpdate
    try:
        if len(search_value) < 3:
            return [{"label": i, "value": i} for i in libraries]
        req_url = "https://libraries.io/api/search?q={lib_name}&sort=stars".format(lib_name=search_value)
        res = get_response(req_url).json()
        new_libraries = [r["name"] for r in res]
        search_url = 'https://libraries.io/api/search?q={}'.format(search_value)
        exact_search = get_response(search_url).json()
        if search_value in [r["name"] for r in exact_search]:
            new_libraries.append(search_value)
        all_libraries = list(set(value + new_libraries))
        all_libraries.sort(key=len)
        all_values = [{"label": r, "value": r} for r in all_libraries]

    except Exception as e:
        print(e)
    return all_values or [{"label": i, "value": i} for i in libraries]

@app.callback(
    Output("measure-select", "value"),
    [Input("measure-select", "options"), Input("measure-select-all", "value")],
)
def update_measure_dropdown(options, select_all):
    if select_all == ["All"]:
        value = [i["value"] for i in options]
    else:
        value = dash.no_update
    return value

# Updates the Select All checkbox as the measure select is updated
@app.callback(
    Output("measure-checklist-container", "children"),
    [Input("measure-select", "value")],
    [State("measure-select", "options"), State("measure-select-all", "value")],
)
def update_measure_checklist(selected, select_options, checked):
    if len(selected) < len(select_options) and len(checked) == 0:
        raise PreventUpdate()

    elif len(selected) < len(select_options) and len(checked) == 1:
        return dcc.Checklist(
            id="measure-select-all",
            options=[{"label": "Select All", "value": "All"}],
            value=[],
        )

    elif len(selected) == len(select_options) and len(checked) == 1:
        raise PreventUpdate()

    return dcc.Checklist(
        id="measure-select-all",
        options=[{"label": "Select All", "value": "All"}],
        value=["All"],
    )

@app.callback(
    Output("libraries-table-container", "children"),
    [
        Input("library-select", "value"),
    ],
)
def update_libraries_table(library_select):
    columns = [
        "name",
        "licenses",
        "latest_release_published_at",
        "latest_release_number",
        "description"
    ]

    data_list = update_libraries(library_select)

    return dash_table.DataTable(
        id="libraries-table",
        columns=[{"name": i, "id": i} for i in columns],
        data= data_list,
        #filter_action="native",
        page_size=10,
        style_cell={"background-color": "#242a3b", "color": "#FFF", "text-align": "left"},
        style_as_list_view=False,
        style_header={"background-color": "#1f2536", "padding": "0px 5px"}
    )


def update_libraries(library_select):
    data_list = []
    for package in library_select:
        data = lib.get_package(package=package)
        data_list.append(data)
    return data_list


@app.callback(
    Output("parallel-coordinates", "figure"),
    [
        Input("library-select", "value"),
        Input("measure-select", "value"),
    ],
)
def update_parallel_coordinates(library_select, measure_select):
    return generate_parallel_coordinates(library_select, measure_select)


@app.callback(
    Output("dependency-graph-container", "children"),
    [
        Input("library-select", 'value'),
    ],
)
def update_dependency_graph(selected_libraries):
    update_libraries(selected_libraries)
    return generate_dependency_graph(
        selected_libraries
    )

if __name__ == "__main__":
    app.run_server(debug=True)