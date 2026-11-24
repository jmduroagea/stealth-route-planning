# Required imports
import numpy as np
from Location import Location
from Boundaries import Boundaries
from Radar import Radar
from tqdm import tqdm

# Constant that avoids setting cells to have an associated cost of zero
EPSILON = 1e-4

class Map:
    """ Class that models the map for the simulation """
    def __init__(self, 
                 boundaries: Boundaries,
                 height:     np.int32, 
                 width:      np.int32, 
                 radars:     np.array=None):
        self.boundaries = boundaries        # Boundaries of the map
        self.height     = height            # Number of coordinates in the y-axis
        self.width      = width             # Number of coordinates int the x-axis
        self.radars     = radars            # List containing the radars (objects)

    def generate_radars(self, n_radars: np.int32) -> None:
        """ Generates n-radars randomly and inserts them into the radars list """
        # Select random coordinates inside the boundaries of the map
        lat_range = np.linspace(start=self.boundaries.min_lat, stop=self.boundaries.max_lat, num=self.height)
        lon_range = np.linspace(start=self.boundaries.min_lon, stop=self.boundaries.max_lon, num=self.width)
        rand_lats = np.random.choice(a=lat_range, size=n_radars, replace=False)
        rand_lons = np.random.choice(a=lon_range, size=n_radars, replace=False)
        self.radars = []        # Initialize 'radars' as an empty list

        # Loop for each radar that must be generated
        for i in range(n_radars):
            # Create a new radar
            new_radar = Radar(location=Location(latitude=rand_lats[i], longitude=rand_lons[i]),
                              transmission_power=np.random.uniform(low=1, high=1000000),
                              antenna_gain=np.random.uniform(low=10, high=50),
                              wavelength=np.random.uniform(low=0.001, high=10.0),
                              cross_section=np.random.uniform(low=0.1, high=10.0),
                              minimum_signal=np.random.uniform(low=1e-10, high=1e-15),
                              total_loss=np.random.randint(low=1, high=10),
                              covariance=None)

            # Insert the new radar
            self.radars.append(new_radar)
        return


    def get_radars_locations_numpy(self) -> np.array:
        """
        Converts radar locations to a NumPy array of geographic coordinates.

        Returns:
            np.array: A 2D array of shape (n_radars, 2) containing the
                     (latitude, longitude) coordinates of each radar.
                     Array dtype is float32 for memory efficiency.
        """
        # Initialize empty array with shape (number_of_radars, 2)
        locations = np.zeros(shape=(len(self.radars), 2), dtype=np.float32)

        # Populate array with radar coordinates
        for i in range(len(self.radars)):
            # Convert each Radar object's location to numpy array
            locations[i] = self.radars[i].location.to_numpy()

        return locations


    def compute_detection_map(self) -> np.array:
        """
        Computes a 2D detection probability map across the entire grid.

        The detection probability at each cell is determined by:
        1. Calculating the maximum detection value from all radars
        2. Scaling values to range [EPSILON, 1] using MinMax normalization

        Returns:
            np.array: A 2D array of shape (height, width) containing
                     scaled detection probabilities (0-1 range with EPSILON offset)

        Process:
            1. Creates grid covering the geographic boundaries
            2. For each cell, computes maximum detection from all radars
            3. Applies MinMax scaling with EPSILON to avoid zero probabilities
        """
        # Initialize detection map with zeros (height x width grid)
        detection_map = np.zeros(shape=(self.height, self.width), dtype=np.float32)

        # Create linear ranges for latitude and longitude coordinates
        # These correspond to the grid rows and columns
        lat_range = np.linspace(self.boundaries.min_lat,
                                self.boundaries.max_lat,
                                self.height)
        lon_range = np.linspace(self.boundaries.min_lon,
                                self.boundaries.max_lon,
                                self.width)

        # Compute detection values for each grid cell
        # tqdm provides progress bar for large grids
        for i in tqdm(range(self.height)):
            for j in range(self.width):
                max_detection = 0.0  # Track maximum detection at this cell

                # Check detection from all radars
                for radar in self.radars:
                    # Get detection probability from current radar
                    detection = radar.compute_detection_level(lat_range[i], lon_range[j])

                    # Keep only the maximum detection value
                    if detection > max_detection:
                        max_detection = detection

                # Store the maximum detection probability for this cell
                detection_map[i, j] = max_detection

        # Scale detection values to [EPSILON, 1] range
        min_val = np.min(detection_map)
        max_val = np.max(detection_map)

        if max_val != min_val:  # Avoid division by zero for uniform maps
            # MinMax scaling with EPSILON offset:
            # 1. Normalize to [0,1] range
            # 2. Scale to [EPSILON, 1] to avoid zero probabilities
            detection_map = ((detection_map - min_val) /
                             (max_val - min_val)) * (1 - EPSILON) + EPSILON
        else:
            # Handle case where all values are identical
            detection_map.fill(EPSILON)  # Set all cells to minimum probability

        return detection_map