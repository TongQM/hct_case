"""
Shared constants used across the transit partitioning project.
"""

# Autonomous shuttles are assumed to cost 5x less per mile than human-driven vehicles.
# When computing provider travel costs for TSP mode, divide the travel distance by this factor.
TSP_TRAVEL_DISCOUNT = 5.0
