# Koda rakstīšanai tika izmantota ChatGPT palīdzība.
import dash
from dash import html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import base64
import io
import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import KDTree
from regions import all_coordinates, region_colors
from regionsId import all_ids

class GraphVisualizer:
    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.triangles = np.loadtxt(np.DataSource().open('https://raw.githubusercontent.com/mingruixia/BrainNet-Viewer/master/Data/SurfTemplate/BrainMesh_Ch2_smoothed.nv'), skiprows=53471, dtype=int)
        self.coords = np.loadtxt(np.DataSource().open('https://raw.githubusercontent.com/mingruixia/BrainNet-Viewer/master/Data/SurfTemplate/BrainMesh_Ch2_smoothed.nv'), skiprows=1, max_rows=53469)
        self.kdtrees = {}
        self.regions = {}
        self.increment = 0
        for region, coords in all_coordinates.items():
            self.kdtrees[region] = KDTree(coords)
        self.init_layout()

    def init_layout(self):
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col(dcc.Upload(
                    id='upload-matrix1',
                    children=dbc.Button('Augšupielādēt matricu 1'),
                    multiple=False
                ), width="auto"),
                dbc.Col(dcc.Upload(
                    id='upload-matrix2',
                    children=dbc.Button('Augšupielādēt matricu 2', id='upload-matrix2-button', disabled=True),
                    multiple=False
                ), width="auto"),
                dbc.Col(html.A(dbc.Button('Lejupielādēt rezultātus 1', id='download-parameters1'), id='download-link1', download="parameters1.txt", href="", target="_blank"), width="auto"),
                dbc.Col(html.A(dbc.Button('Lejupielādēt rezultātus 2', id='download-parameters2', disabled=True), id='download-link2', download="parameters2.txt", href="", target="_blank"), width="auto"),
                dbc.Col(html.A(dbc.Button('Lejupielādēt visus rezultātus', id='download-parameters-all', disabled=True), id='download-link-all', download="all_parameters.txt", href="", target="_blank"), width="auto"),
                dbc.Col(html.Button('Salīdzināt', id='compare', disabled=True), width="auto")
            ], justify="start"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='graph-output1')
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='graph-output2')
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='3d-brain-graph1')
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='3d-brain-graph2')
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dash_table.DataTable(id='graph-parameters1', columns=[{"name": "Parametrs", "id": "Parameter"},{"name": "Vērtība", "id": "Value"} ], data=[])
                ]),
                dbc.Col([
                    dash_table.DataTable(id='graph-parameters2', columns=[{"name": "Parametrs", "id": "Parameter"},{"name": "Vērtība", "id": "Value"} ], data=[])
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dash_table.DataTable(id='graph-regions1', columns=[{"name": "Regions", "id": "Regions"}], data=[])
                ]),
                dbc.Col([
                    dash_table.DataTable(id='graph-regions2', columns=[{"name": "Regions", "id": "Regions"}], data=[])
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dash_table.DataTable(id='graph-missing-regions1', columns=[{"name": "Zaudētais regions", "id": "MissingRegions"}], data=[])
                ]),
                dbc.Col([
                    dash_table.DataTable(id='graph-missing-regions2', columns=[{"name": "Zaudētais regions", "id": "MissingRegions"}], data=[])
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dash_table.DataTable(id='parameter-differences-table', columns=[{"name": "Parametrs", "id": "Parameter"}, {"name": "Starpība", "id": "Difference"}], data=[])
                ])
            ]),
        ], fluid=True)

    def mesh_properties(self, mesh_coords):
        radii = []
        center = []
        for coords in mesh_coords:
            c_max = max(coords)
            c_min = min(coords)
            center.append((c_max + c_min) / 2)
            radii.append((c_max - c_min) / 2)
        return center, max(radii)

    def is_point_in_region(self, x, y, z):
        min_distance = float('inf')
        assigned_region = 'Outside Brain Regions'
        for region, kdtree in self.kdtrees.items():
            distance, _ = kdtree.query([x, y, z])
            if distance < min_distance:
                min_distance = distance
                assigned_region = region
        return assigned_region != 'Outside Brain Regions', assigned_region
    
    def brainViz(self, G):        
        x, y, z = self.coords.T
        triangles_zero_offset = self.triangles - 1
        i, j, k = triangles_zero_offset.T

        fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z,
                                        i=i, j=j, k=k,
                                        color='lightpink', opacity=0.5, name='', showscale=False, hoverinfo='none')])
        
        mesh_coords = (x, y, z)
        mesh_center, mesh_radius = self.mesh_properties(mesh_coords)

        scale_factor = 5
        pos_3d = nx.kamada_kawai_layout(G, dim=3, center=mesh_center, scale=scale_factor*mesh_radius) 

        pos_brain = {}

        for node, position in pos_3d.items():
            squared_dist_matrix = np.sum((self.coords - position) ** 2, axis=1)
            pos_brain[node] = self.coords[np.argmin(squared_dist_matrix)]

        nodes_x = {'Outside Brain Regions':[]}  
        nodes_y = {'Outside Brain Regions':[]}
        nodes_z = {'Outside Brain Regions':[]}
        
        for region in all_coordinates:
            nodes_x[region] = []
            nodes_y[region] = []
            nodes_z[region] = []

        for node in pos_brain:
            pos = pos_brain[node]
            is_in_region, lobe = self.is_point_in_region(*pos)
            if is_in_region:
                nodes_x[lobe].append(pos[0])
                nodes_y[lobe].append(pos[1])
                nodes_z[lobe].append(pos[2])
            else:
                nodes_x['Outside Brain Regions'].append(pos[0])
                nodes_y['Outside Brain Regions'].append(pos[1])
                nodes_z['Outside Brain Regions'].append(pos[2])
        edge_x = []
        edge_y = []
        edge_z = []
        for s, t in G.edges():
            edge_x += [pos_brain[s][0], pos_brain[t][0], None]
            edge_y += [pos_brain[s][1], pos_brain[t][1], None]
            edge_z += [pos_brain[s][2], pos_brain[t][2], None]

        for region, color in region_colors.items():
            fig.add_trace(go.Scatter3d(x=nodes_x[region], y=nodes_y[region], z=nodes_z[region],
                                       mode='markers', 
                                       name=region,
                                       marker=dict(size=5, color=color)))

        fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z,
                                   mode='lines',
                                   name='Edges',
                                   opacity=0.1, 
                                   line=dict(color='gray')))

        fig.update_scenes(xaxis_visible=False,
                          yaxis_visible=False,
                          zaxis_visible=False)

        fig.update_layout(autosize=False,
                          width=800,
                          height=800)
        return fig

    def calculate_graph_parameters(self, G):
        try:
            degrees = dict(G.degree())
            degree_values = list(degrees.values())
            parameters = {
                "Tīkla vidēja pakāpe":       np.mean(degree_values),
                "Pakāpju centralitāte":      np.mean(list(nx.degree_centrality(G).values())),
                "Tuvuma centralitāte":   np.mean(list(nx.closeness_centrality(G).values())),
                "Starpības centralitāte": np.mean(list(nx.betweenness_centrality(G).values())),
                "Klasterizācijas koeficients" : np.mean(list(nx.clustering(G).values())),
                "Raksturīgais ceļa garums":     nx.average_shortest_path_length(G),
                "Blīvums":                        nx.density(G)
            }
            parameters = {k: f"{v:.5f}" for k, v in parameters.items()}
        except nx.NetworkXError as e:
            parameters = str(e)
        return parameters

    def download_parameters(self, parameters_text1, parameters_text2, parameter_differences):
        parameters_text1 = '\n'.join([f'Matrix 1 - {key}: {value}' for key, value in parameters_text1.items()])
        parameters_text2 = '\n'.join([f'Matrix 2 - {key}: {value}' for key, value in parameters_text2.items()])
        parameter_differences_text = '\n'.join([f'Difference in {item["Parameter"]}: {item["Difference"]}' for item in parameter_differences])
        return f'{parameters_text1}\n\n{parameters_text2}\n\n{parameter_differences_text}'

    def visualize_2d_graph(self, G):
        pos = nx.spring_layout(G)

        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers',
            hoverinfo='text',
            marker=dict(showscale=True, colorscale='YlGnBu', size=10, color=[], colorbar=dict(thickness=15))
        )
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'))
        return go.Figure(data=[edge_trace, node_trace])

    def defineRegions(self, node_types, graph_index):
        regions = set()
        for node in node_types:
            for region, ids in all_ids.items():
                if node in ids:
                    regions.add(region)
                    break

        self.regions[graph_index] = list(regions)
        return self.regions[graph_index]
    
    def compare_graphs(self, index):
        regions_G1 = self.regions[index]

        regions_G1_set = set(regions_G1)
        all_ids_set = set(all_ids.keys())

        missing_regions_G1 = list(all_ids_set - regions_G1_set)

        return missing_regions_G1

    def calculate_parameter_differences(self, params1, params2):
        differences = []
        for key in params1:
            if key in params2:
                diff = float(params1[key]) - float(params2[key])
                differences.append({"Parameter": key, "Difference": f"{diff:.5f}"})
        return differences
    
    def run(self):
        self.G1 = None
        self.G2 = None
        self.parameters1 = {}
        self.parameters2 = {}

        @self.app.callback(
            [Output('graph-output1', 'figure'),
            Output('3d-brain-graph1', 'figure'),
            Output('graph-parameters1', 'data'),
            Output('graph-regions1', 'data'),
            Output('graph-missing-regions1', 'data'),
            Output('upload-matrix2-button', 'disabled'),
            Output('download-link1', 'href')],
            Input('upload-matrix1', 'contents')
        )
        def update_output1(contents):
            if contents is None:
                return go.Figure(), go.Figure(), [], [], [], True, ""

            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), header=None)
            self.G1 = nx.from_pandas_adjacency(df)
            graph_2d = self.visualize_2d_graph(self.G1)

            regions = self.defineRegions(df.iloc[0].to_list(), 0)
            regions_display = [{"Regions": region} for region in regions]
            self.parameters1 = self.calculate_graph_parameters(self.G1)
            parameters_display = [{"Parameter": key, "Value": value} for key, value in self.parameters1.items()]
            parameters_file_content1 = self.download_parameters(self.parameters1, {}, [])
            parameters_href1 = f"data:text/plain;charset=utf-8,{parameters_file_content1}"

            missing_regions_G1 = self.compare_graphs(0)
            missing_regions_display_G1 = [{"MissingRegions": region} for region in missing_regions_G1]

            return graph_2d, self.brainViz(self.G1), parameters_display, regions_display, missing_regions_display_G1, False, parameters_href1


        @self.app.callback(
            [Output('graph-output2', 'figure'),
            Output('3d-brain-graph2', 'figure'),
            Output('graph-parameters2', 'data'),
            Output('graph-regions2', 'data'),
            Output('graph-missing-regions2', 'data'),
            Output('download-link2', 'href'),
            Output('compare', 'disabled'),
            Output('download-parameters2', 'disabled')],
            Input('upload-matrix2', 'contents')
        )
        def update_output2(contents):
            if contents is None:
                return go.Figure(), go.Figure(), [], [], [], "", True, True

            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), header=None)
            self.G2 = nx.from_pandas_adjacency(df)
            graph_2d = self.visualize_2d_graph(self.G2)

            self.parameters2 = self.calculate_graph_parameters(self.G2)
            parameters_display = [{"Parameter": key, "Value": value} for key, value in self.parameters2.items()]
            regions = self.defineRegions(df.iloc[0].to_list(), 1)
            regions_display = [{"Regions": region} for region in regions]
            self.calculate_parameter_differences(self.parameters1, self.parameters2)
            parameters_file_content2 = self.download_parameters({}, self.parameters2, [])
            parameters_href2 = f"data:text/plain;charset=utf-8,{parameters_file_content2}"
            
            missing_regions_G2 = self.compare_graphs(1)
            missing_regions_display_G2 = [{"MissingRegions": region} for region in missing_regions_G2]

            return graph_2d, self.brainViz(self.G2), parameters_display, regions_display, missing_regions_display_G2, parameters_href2, False, False


        @self.app.callback(
            [Output('parameter-differences-table', 'data'),
            Output('download-link-all', 'href'),
            Output('download-parameters-all', 'disabled')],
            Input('compare', 'n_clicks')
        )
        def compare_graphs_callback(n_clicks):
            if n_clicks is None or self.G1 is None or self.G2 is None:
                return [], "", True

            parameter_differences = self.calculate_parameter_differences(self.parameters1, self.parameters2)
            parameters_file_content_all = self.download_parameters(self.parameters1, self.parameters2, parameter_differences)
            parameters_href_all = f"data:text/plain;charset=utf-8,{parameters_file_content_all}"

            return parameter_differences, parameters_href_all, False

        self.app.run_server(debug=True)



if __name__ == '__main__':
    graph_visualizer = GraphVisualizer()
    graph_visualizer.run()
