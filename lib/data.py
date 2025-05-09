import geopandas as gpd
import numpy as np
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
