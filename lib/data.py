import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from libpysal.weights import Queen


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
            self.gdf = self.gdf[self.gdf['GEOIDFQ'].isin(geoid_list)]
        
        elif level == 'block_group':
            self.gdf = gpd.read_file(filepath)
            self.gdf["short_GEOID"] = self.gdf["GEOID"].str[-7:]  # Create a short GEOID column, which also uniquely identifies each block group
            self.gdf.set_index("short_GEOID", inplace=True)  # Set the short GEOID as the index for easier access
            self.short_geoid_list = [geoid[-7:] for geoid in geoid_list]
            # Retrieve block groups with the specified GEOIDs
            self.gdf = self.gdf.loc[self.short_geoid_list]

        elif level == 'block':
            self.gdf = gpd.read_file(filepath)
            self.gdf = self.gdf[self.gdf["GEOIDFQ20"].str.startswith(tuple(geoid_list))]


        if self.gdf.crs.is_geographic:
            self.gdf = self.gdf.to_crs(epsg=2163)
        
        # Calculate the area for each tract
        self.gdf['area'] = self.gdf.geometry.area / 1e6 # Convert to square kilometers

        # Compute contiguity weights
        w = Queen.from_dataframe(self.gdf, use_index=False)
        # Build graph using indices from gdf_subset
        self.G = nx.Graph()
        for tract, neighbors in w.neighbors.items():
            for neighbor in neighbors:
                self.G.add_edge(tract, neighbor)

        # Use centroids of each part for node positions
        self.pos = {i: (geom.centroid.x, geom.centroid.y) for i, geom in enumerate(self.gdf.geometry)}

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

    def generate_demand_dist(self, commuting_df, population_df):
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

class DemandData:
    def __init__(self, meta_path, data_path, level='tract', datatype='commuting'):
        """
        Load data from a CSV file and rename columns based on metadata.
        
        Parameters:
        - meta_path: Path to the metadata CSV file.
        - data_path: Path to the data CSV file.
        
        Returns:
        - DataFrame with renamed columns.
        """
        assert datatype in ['commuting', 'population'], "Invalid datatype. Choose 'commuting' or 'population'."
        if datatype == 'commuting' and level != 'tract':
            raise ValueError("Commuting data is only available at the tract level.")
        
        self.level = level
        meta = pd.read_csv(meta_path)
        code_to_label = dict(zip(meta["Column Name"], meta["Label"]))
        
        self.data = pd.read_csv(data_path)
        self.data.rename(columns=code_to_label, inplace=True)
        
        # Remove "!!" from column names
        self.data.columns = self.data.columns.str.replace("!!", " ", regex=False)


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
