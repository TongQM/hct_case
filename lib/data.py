import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
from libpysal.weights import Queen
import polyline
import json
from shapely.geometry import LineString, Point
from shapely.ops import transform
from pyproj import Transformer


class GeoData:
    def __init__(self, filepath, geoid_list, level='tract'):
        """
        Initialize the geoData object.

        Parameters:
        - filepath: str, path to the shapefile
        - geoid_list: list, list of geoid values to filter
        - level: str, level of geography (default is 'tract')
        """
        self.filepath = filepath
        self.geoid_list = geoid_list
        self.level = level
        if level == 'tract':
            self.gdf = gpd.read_file(filepath)
            self.gdf["short_GEOID"] = self.gdf["GEOID"].str[-6:]
            self.gdf.set_index("short_GEOID", inplace=True)
            self.short_geoid_list = [geoid[-6:] for geoid in geoid_list]
            # Retrieve tracts with the specified GEOIDs
            self.gdf = self.gdf.loc[self.short_geoid_list]            
        
        elif level == 'block_group':
            self.gdf = gpd.read_file(filepath)
            self.gdf["short_GEOID"] = self.gdf["GEOID"].str[-7:]  # Create a short GEOID column, which also uniquely identifies each block group
            self.gdf.set_index("short_GEOID", inplace=True)  # Set the short GEOID as the index for easier access
            self.short_geoid_list = [geoid[-7:] for geoid in geoid_list]
            # Retrieve block groups with the specified GEOIDs
            self.gdf = self.gdf.loc[self.short_geoid_list]

        elif level == 'block':
            self.gdf = gpd.read_file(filepath)
            self.gdf["short_GEOID"] = self.gdf["GEOID20"].str[-10:]
            tract_short_geoid_list = [geoid[-6:] for geoid in geoid_list]
            self.gdf = self.gdf[self.gdf["short_GEOID"].str.startswith(tuple(tract_short_geoid_list))]
            self.gdf.set_index("short_GEOID", inplace=True)
            self.short_geoid_list = self.gdf.index.tolist()


        if self.gdf.crs.is_geographic:
            self.gdf = self.gdf.to_crs(epsg=2163)
        
        # Calculate the area for each geometry in square kilometers
        # Note: The area is calculated in the projected coordinate system (EPSG:2163)
        self.gdf['area'] = self.gdf.geometry.area / 1e6 # Convert to square kilometers

        # Compute contiguity weights
        w = Queen.from_dataframe(self.gdf, use_index=True)
        # Build graph using indices from gdf_subset
        self.G = nx.Graph()
        for node, neighbors in w.neighbors.items():
            for neighbor in neighbors:
                self.G.add_edge(node, neighbor)

        # Use centroids of each part for node positions
        self.pos = {short_geoid: (geom.centroid.x, geom.centroid.y) for short_geoid, geom in self.gdf.geometry.items()}
        # self.pos = {i: (geom.centroid.x, geom.centroid.y) for i, geom in enumerate(self.gdf.geometry)}

        # Compute Euclidean distances between centroids for each edge
        self.edge_labels = {}
        self.distance_dict = {}
        for (node1, node2) in self.G.edges():
            pt1 = self.pos[node1]
            pt2 = self.pos[node2]
            distance = ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5 / 1000  # in kilometers
            self.G[node1][node2]['distance'] = distance  # Store distance in the graph
            self.edge_labels[(node1, node2)] = f"{distance:.2f} km"
            self.distance_dict[(node1, node2)] = distance

        # Compute shortest path lengths for all pairs using Dijkstra
        shortest_paths = dict(nx.all_pairs_dijkstra_path_length(self.G, weight='distance'))

        # Flatten the result into a dictionary with keys as (node1, node2)
        self.shortest_distance_dict = {}
        for src, paths in shortest_paths.items():
            for tgt, dist in paths.items():
                self.shortest_distance_dict[(src, tgt)] = dist

    def get_area(self, bg_geoid):
        return self.gdf['area'][bg_geoid]
    
    def get_dist(self, node1, node2):
        return self.shortest_distance_dict[(node1, node2)]

    def generate_demand_dist(self, commuting_df, population_df):
        '''
        Legacy code for generating demand distribution based on population and commuting data.
        This function is not used in the current implementation.
        It is recommended to construct the probaility distribution based on the demand data directly.
        '''
        # Population ratio manipulation
        selected_columns = ['Geography', 'Geographic Area Name', 'Estimate Total:']
        block_group_pop_data = population_df[selected_columns]
        new_column_names = ['GEOID', 'block_group_name', 'total_population']
        block_group_pop_data.columns = new_column_names
        tract_population = (
            block_group_pop_data
            .groupby('short_tract_GEOID', as_index=False)['total_population']
            .sum()
            .rename(columns={'total_population': 'tract_population'})
        )
        block_group_pop_data = block_group_pop_data.merge(tract_population, on='short_tract_GEOID', how='left')
        block_group_pop_data['population%'] = block_group_pop_data['total_population'] / block_group_pop_data['tract_population'] * 100

        # Commuting demand estimation proportional to population
        commuting_df = commuting_df.rename(columns={'GEOID': 'block_group_name'})
        commuting_df = commuting_df.merge(block_group_pop_data[['block_group_name', 'population%']], on='block_group_name', how='left')
        commuting_df['commuting_demand'] = commuting_df['population%'] * commuting_df['total_commuting']
        

    def plot_graph(self, savepath='./figures/'):
        # Plot the tracts and overlay the graph with distance labels
        fig, ax = plt.subplots(figsize=(30, 30))
        self.gdf.plot(ax=ax, color='lightgrey', edgecolor='black')
        if self.level == 'block':
            nx.draw(self.G, self.pos, ax=ax, node_color='red', edge_color='blue', with_labels=False, node_size=1)
        else:
            nx.draw(self.G, self.pos, ax=ax, node_color='red', edge_color='blue', with_labels=True, node_size=1)
            nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=self.edge_labels, font_color='green', font_size=2)
        plt.savefig(f"{savepath}{self.level}_graph.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


    def plot_partition(self, assignment):
        """
        Visualize the partition of block groups into districts.
        
        Parameters:
        assignment (np.ndarray): Binary array of shape (n_block_groups, n_centers),
                                where each row has exactly one 1.
        gdf (GeoDataFrame): GeoDataFrame of block groups; order must match assignment rows.
        
        Returns:
        centers (dict): Dictionary mapping each district (center index) to its center block group id.
        """
        # Convert binary assignment to a district label per block group.
        district_labels = np.argmax(assignment, axis=1)
        gdf = self.gdf.copy()  # avoid modifying the original GeoDataFrame
        gdf['district'] = district_labels

        # Create the plot with a categorical colormap.
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf.plot(column='district', cmap='tab20', legend=True, ax=ax)
        
        centers = {}
        # For each district, determine the center block group.
        for district in np.unique(district_labels):
            subset = gdf[gdf['district'] == district]
            # Compute centroids of the block groups in this district.
            centroids = subset.geometry.centroid
            # Compute the average centroid (district centroid)
            avg_x = centroids.x.mean()
            avg_y = centroids.y.mean()
            # Find the block group whose centroid is closest to the district centroid.
            distances = centroids.apply(lambda geom: ((geom.x - avg_x)**2 + (geom.y - avg_y)**2)**0.5)
            center_idx = distances.idxmin()  # center's index (e.g., GEOID)
            centers[district] = center_idx
            
            # # Plot the boundary of the center block group with a thicker line.
            # subset.loc[[center_idx]].boundary.plot(ax=ax, edgecolor='black', linewidth=3)
            # Optionally, add a marker at the center.
            # ax.plot(centroids.loc[center_idx].x, centroids.loc[center_idx].y, marker='o',
            #         color='white', markersize=3)
            ax.plot(avg_x, avg_y, marker='o',
                    color='white', markersize=3)
        
        plt.title("District Partition with Center Block Groups")
        plt.show()
        
        return centers
 

class RouteData:
    def __init__(self, data_path, geodata: GeoData):

        assert data_path.endswith('.json'), "Data path must be a JSON file."

        # Load the routes info from a JSON file
        with open(data_path, "r") as f:
            self.routes_info = json.load(f)

        self.geodata = geodata
        self.short_geoid_list = geodata.short_geoid_list
        self.level = geodata.level
        self.gdf = geodata.gdf

        # Create a transformer from EPSG:4326 (lat/lon) to EPSG:2163 (projected in meters)
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:2163", always_xy=True)

    def project_geometry(self, geometry):
        """Project a shapely geometry from EPSG:4326 to EPSG:2163."""
        return transform(self.transformer.transform, geometry)

    def visualize_routes_and_nodes(self):
        """
        Visualize routes overlaid on level nodes (projected in EPSG:2163),
        and return a dictionary mapping each route name to a list of stop info.
        
        Parameters:
        gdf (GeoDataFrame): Node polygons (CRS EPSG:2163).
        routes_info (list): List of route dictionaries, each with keys:
            - "Description": route name.
            - "EncodedPolyline": encoded polyline string.
            - "MapLineColor": color for plotting.
            - "Stops": list of stop dictionaries with keys "Latitude", "Longitude", "Description".
            
        Returns:
            dict: Keys are route names, and values are lists of stop info dictionaries.
        """
        stops_dict = {}
        
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # Plot block groups as the base layer (already in EPSG:2163)
        self.gdf.plot(ax=ax, color='lightgrey', edgecolor='black')
        
        for route in self.routes_info:
            route_name = route.get("Description", "Unnamed Route")
            encoded_poly = route.get("EncodedPolyline")
            color = route.get("MapLineColor", "#000000")
            stops = route.get("Stops", [])
            
            # Decode the polyline (originally in lat/lon) and project to EPSG:2163.
            if encoded_poly:
                try:
                    # Decode returns (lat, lng) pairs. Swap to (lng, lat) for shapely.
                    coords = polyline.decode(encoded_poly)
                    coords_swapped = [(lng, lat) for lat, lng in coords]
                    route_line = LineString(coords_swapped)
                    projected_line = self.project_geometry(route_line)
                    ax.plot(*projected_line.xy, color=color, linewidth=2, label=route_name)
                except Exception as e:
                    print(f"Error decoding polyline for route '{route_name}': {e}")
            
            route_stops = []
            for stop in stops:
                lat = stop.get("Latitude")
                lng = stop.get("Longitude")
                desc = stop.get("Description", "")
                route_stops.append({"Latitude": lat, "Longitude": lng, "Description": desc})
                
                # Create a Point and project it
                pt = Point(lng, lat)
                projected_pt = self.project_geometry(pt)
                ax.scatter(projected_pt.x, projected_pt.y, color=color, s=50, edgecolor='k', zorder=5)
                ax.text(projected_pt.x, projected_pt.y, desc, fontsize=8, color=color)
            
            stops_dict[route_name] = route_stops

        ax.set_title("Routes and Block Groups (EPSG:2163 Projection)")
        ax.legend()
        plt.show()
        
        return stops_dict


    def _get_fixed_route_assignment(self, visualize=True):
        """
        Partition nodes into Districts based on the nearest stop, then plot with proper legend.
        """
        # Assign each block group to nearest route
        district_assignment = {}
        for idx, row in self.gdf.iterrows():
            centroid = row.geometry.centroid
            best_route, best_dist = None, float('inf')
            for route in self.routes_info:
                name = route.get("Description", f"Route {route.get('RouteID')}")
                for stop in route.get("Stops", []):
                    pt = Point(stop['Longitude'], stop['Latitude'])
                    pt_proj = self.project_geometry(pt)
                    d = centroid.distance(pt_proj)
                    if d < best_dist:
                        best_dist, best_route = d, name
            district_assignment[idx] = best_route
        self.gdf['district'] = self.gdf.index.map(district_assignment)

        # Find center of each district
        center_nodes = {}
        for district, group in self.gdf.groupby('district'):
            centroids = group.geometry.centroid
            mean_point = Point(centroids.x.mean(), centroids.y.mean())
            best_idx, best_dist = None, float('inf')
            for idx, geom in group.geometry.items():
                d = geom.centroid.distance(mean_point)
                if d < best_dist:
                    best_dist, best_idx = d, idx
            center_nodes[district] = best_idx
        self.center_nodes = center_nodes


        if visualize:
            # Plotting
            unique_districts = self.gdf['district'].unique()
            cmap = cm.get_cmap('Set1', len(unique_districts))
            color_map = {d: cmap(i) for i, d in enumerate(unique_districts)}

            fig, ax = plt.subplots(figsize=(15, 15))
            for district in unique_districts:
                subset = self.gdf[self.gdf['district'] == district]
                subset.plot(ax=ax, color=color_map[district], edgecolor='black')

            # Overlay routes
            for route in self.routes_info:
                name = route.get("Description", f"Route {route.get('RouteID')}")
                poly = route.get("EncodedPolyline")
                color = route.get("MapLineColor", '#000000')
                if poly:
                    try:
                        coords = polyline.decode(poly)
                        line = LineString([(lng, lat) for lat, lng in coords])
                        proj_line = self.project_geometry(line)
                        ax.plot(*proj_line.xy, color=color, linewidth=2)
                    except Exception as e:
                        print(f"Error decoding polyline for {name}: {e}")

            # Plot centers
            for district, idx in center_nodes.items():
                geom = self.gdf.loc[idx].geometry
                boundary = getattr(geom, 'exterior', geom.boundary)
                ax.plot(*boundary.xy, color='red', linewidth=3)
                pt = geom.centroid
                ax.plot(pt.x, pt.y, 'o', color='red', markersize=10)
                ax.text(pt.x, pt.y, str(district), fontsize=12, fontweight='bold',
                        color='red', ha='center', va='center')

            # Build custom legend for districts
            legend_handles = [
                mpatches.Patch(facecolor=color_map[d], edgecolor='black', label=str(d))
                for d in unique_districts
            ]
            ax.legend(handles=legend_handles, title='District', loc='upper right')
            plt.title("Partition of Block Groups into Districts by Nearest Route Stop")
            plt.show()

    def build_assignment_matrix(self, visualize=True):
        """
        Construct a binary assignment matrix where entry (i, j) = 1 if the i-th block group
        (in self.short_geoid_list) is assigned to the center corresponding to the j-th center.

        Returns:
            assignment: np.ndarray of shape (n_groups, n_centers)
            center_list: list of center node identifiers (short GEOID index)
        """
        
        self._get_fixed_route_assignment(visualize=visualize)
        center_list = list(self.center_nodes.values())
        n_groups = len(self.short_geoid_list)
        n_centers = len(center_list)
        assignment = np.zeros((n_groups, n_centers), dtype=int)

        # Map geoid to row index
        geo_to_row = {geoid: i for i, geoid in enumerate(self.short_geoid_list)}
        # Map center idx to column
        center_to_col = {center: j for j, center in enumerate(center_list)}

        for geoid in self.short_geoid_list:
            i = geo_to_row[geoid]
            district = self.gdf.loc[geoid, 'district']
            center = self.center_nodes[district]
            j = center_to_col[center]
            assignment[i, j] = 1

        return assignment, center_list


    def evaluate_partition(self, assignment, mode="fixed"):
        """
        Evaluate the partition of nodes into districts.
        
        Parameters:
        assignment (np.ndarray): Binary array of shape (n_block_groups, n_centers),
                                where each row has exactly one 1.
        mode (str): Mode of evaluation, either "fixed" or "tsp".
        
        Returns:
        centers (dict): Dictionary mapping each district (center index) to its center block group id.
        """
        assert mode in ["fixed", "tsp"], "Invalid mode. Choose 'fixed' or 'tsp'."

        # Convert binary assignment to a district label per block group.
        district_labels = np.argmax(assignment, axis=1)
        gdf = self.gdf.copy()



# class DemandData:
#     def __init__(self, meta_path, data_path, level='tract', datatype='commuting'):
#         """
#         Load data from a CSV file and rename columns based on metadata.
        
#         Parameters:
#         - meta_path: Path to the metadata CSV file.
#         - data_path: Path to the data CSV file.
        
#         Returns:
#         - DataFrame with renamed columns.
#         """
#         assert datatype in ['commuting', 'population'], "Invalid datatype. Choose 'commuting' or 'population'."
#         if datatype == 'commuting' and level != 'tract':
#             raise ValueError("Commuting data is only available at the tract level.")
        
#         self.level = level
#         meta = pd.read_csv(meta_path)
#         code_to_label = dict(zip(meta["Column Name"], meta["Label"]))
        
#         self.data = pd.read_csv(data_path)
#         self.data.rename(columns=code_to_label, inplace=True)
        
#         # Remove "!!" from column names
#         self.data.columns = self.data.columns.str.replace("!!", " ", regex=False)


def load_data(meta_path, data_path):
    """
    Load data from a CSV file and rename columns based on metadata.
    
    Parameters:
    - meta_path: Path to the metadata CSV file.
    - data_path: Path to the data CSV file.
    
    Returns:
    - DataFrame with renamed columns.
    """
    meta = pd.read_csv(meta_path)
    code_to_label = dict(zip(meta["Column Name"], meta["Label"]))
    
    data = pd.read_csv(data_path)
    data.rename(columns=code_to_label, inplace=True)
    
    # Remove "!!" from column names
    data.columns = data.columns.str.replace("!!", " ", regex=False)
    
    return data
