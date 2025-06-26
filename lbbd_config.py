"""
Configuration file for LBBD experiments
Modify this file to customize your experiments
"""

import os

# Data paths
DATA_PATHS = {
    'shapefile': "2023_shape_files/census_shape_block_group/tl_2023_42_bg.shp",
    'commuting_data': '2023_target_tract_commuting/ACSDT5Y2023.B08303-Data.csv',
    'population_data': '2023_target_blockgroup_population/ACSDT5Y2023.B01003-Data.csv'
}

# Block group GEOID list - Updated to match data_process.ipynb
BG_GEOID_LIST = [
    "1500000US420034980001", "1500000US420034980002", "1500000US420034993001", "1500000US420034993002", "1500000US420034994001",
    "1500000US420034994002", "1500000US420034994003", "1500000US420035003001", "1500000US420035003002", "1500000US420035003003",
    "1500000US420035003004", "1500000US420035010001", "1500000US420035030021", "1500000US420035030022", "1500000US420035030023",
    "1500000US420035030024", "1500000US420035030025", "1500000US420035041001", "1500000US420035041002", "1500000US420035041003",
    "1500000US420035041004", "1500000US420035041005", "1500000US420035070001", "1500000US420035070002", "1500000US420035080001",
    "1500000US420035080002", "1500000US420035094001", "1500000US420035094002", "1500000US420035094003", "1500000US420035094004",
    "1500000US420035094005", "1500000US420035100001", "1500000US420035100002", "1500000US420035120001", "1500000US420035120002",
    "1500000US420035130001", "1500000US420035130002", "1500000US420035138001", "1500000US420035138002", "1500000US420035140001",
    "1500000US420035140002", "1500000US420035151001", "1500000US420035151002", "1500000US420035151003", "1500000US420035152001",
    "1500000US420035152002", "1500000US420035153001", "1500000US420035153002", "1500000US420035154011", "1500000US420035154012",
    "1500000US420035154013", "1500000US420035161001", "1500000US420035162001", "1500000US420035162002", "1500000US420035170001",
    "1500000US420035170002", "1500000US420035180011", "1500000US420035180012", "1500000US420035180013", "1500000US420035190001",
    "1500000US420035190002", "1500000US420035190003", "1500000US420035200011", "1500000US420035200012", "1500000US420035200021",
    "1500000US420035200022", "1500000US420035200023", "1500000US420035212001", "1500000US420035212002", "1500000US420035212003",
    "1500000US420035213011", "1500000US420035213012", "1500000US420035213013", "1500000US420035213014", "1500000US420035213021",
    "1500000US420035213022", "1500000US420035213023", "1500000US420035213024", "1500000US420035214011", "1500000US420035214012",
    "1500000US420035214021", "1500000US420035214022", "1500000US420035214023", "1500000US420035220001", "1500000US420035220002",
    "1500000US420035220003", "1500000US420035509001", "1500000US420035509002", "1500000US420035512001", "1500000US420035512002",
    "1500000US420035512003", "1500000US420035513001", "1500000US420035513002", "1500000US420035519001", "1500000US420035520001",
    "1500000US420035520002", "1500000US420035520003", "1500000US420035520004", "1500000US420035521001", "1500000US420035521002",
    "1500000US420035522001", "1500000US420035523001", "1500000US420035523002", "1500000US420035523003", "1500000US420035524001",
    "1500000US420035524002", "1500000US420035524003", "1500000US420035614001", "1500000US420035614002", "1500000US420035614003",
    "1500000US420035614004", "1500000US420035615001", "1500000US420035615002", "1500000US420035639001", "1500000US420035639002",
    "1500000US420035639003", "1500000US420035639004", "1500000US420035642001", "1500000US420035642002", "1500000US420035642003",
    "1500000US420035644001", "1500000US420035644002", "1500000US420035644003", "1500000US420035644004", "1500000US420035644005",
    "1500000US420035644006", "1500000US420035644007", "1500000US420035647001", "1500000US420035647002", "1500000US420035647003"
]

# Experiment configurations
LBBD_CONFIGS = [
    {
        'name': 'tiny_test',
        'description': 'Tiny test with just 6 blocks for debugging',
        'num_districts': 2,
        'epsilon': 0.1,
        'max_iterations': 5,
        'tolerance': 1e-2,
        'max_cuts': 10,
        'verbose': True
    },
    {
        'name': 'quick_test',
        'description': 'Quick test with minimal iterations',
        'num_districts': 3,
        'epsilon': 0.1,
        'max_iterations': 10,
        'tolerance': 1e-2,
        'max_cuts': 20,
        'verbose': True
    },
    {
        'name': 'small_scale',
        'description': 'Small scale experiment',
        'num_districts': 3,
        'epsilon': 0.1,
        'max_iterations': 20,
        'tolerance': 1e-3,
        'max_cuts': 50,
        'verbose': True
    },
    {
        'name': 'standard',
        'description': 'Standard LBBD run',
        'num_districts': 3,
        'epsilon': 0.05,
        'max_iterations': 50,
        'tolerance': 1e-4,
        'max_cuts': 100,
        'verbose': True
    },
    {
        'name': 'high_precision',
        'description': 'High precision with tight tolerance',
        'num_districts': 3,
        'epsilon': 0.01,
        'max_iterations': 100,
        'tolerance': 1e-5,
        'max_cuts': 200,
        'verbose': True
    },
    {
        'name': 'five_districts',
        'description': 'Test with 5 districts',
        'num_districts': 5,
        'epsilon': 0.05,
        'max_iterations': 50,
        'tolerance': 1e-4,
        'max_cuts': 100,
        'verbose': True
    }
]

# Small GEOID list for debugging
SMALL_BG_GEOID_LIST = [
    "1500000US420034980001", "1500000US420034980002", "1500000US420034993001", 
    "1500000US420034993002", "1500000US420034994001", "1500000US420034994002"
]

# Output settings
OUTPUT_SETTINGS = {
    'results_dir': 'lbbd_results',
    'log_level': 'INFO',
    'save_plots': True,
    'save_detailed_history': True
}

# Computational settings
COMPUTATION_SETTINGS = {
    'random_seed': 42,
    'parallel_processing': False,
    'memory_limit_gb': 8,
    'time_limit_hours': 2
}

def get_config_by_name(name):
    """Get configuration by name"""
    for config in LBBD_CONFIGS:
        if config['name'] == name:
            return config
    raise ValueError(f"Configuration '{name}' not found")

def list_available_configs():
    """List all available configurations"""
    print("Available LBBD configurations:")
    print("-" * 50)
    for config in LBBD_CONFIGS:
        print(f"Name: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"Districts: {config['num_districts']}, Epsilon: {config['epsilon']}")
        print(f"Max iterations: {config['max_iterations']}, Tolerance: {config['tolerance']}")
        print("-" * 50)

if __name__ == "__main__":
    list_available_configs() 