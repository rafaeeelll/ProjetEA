"""
Interactive atmosphere density sphere with species selection and altitude slider.
The sphere radius is modulated by atmospheric density of selected species.
"""

import numpy as np
import plotly.graph_objects as go
from nrlmsise00 import msise_model
from datetime import datetime


def spherical_to_cartesian(radius, lat_deg, lon_deg):
    """Convert spherical coordinates to Cartesian."""
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    
    return x, y, z


def get_species_density(alt_km, lat_deg, lon_deg, species='N2', year=2020):
    """Get density of specific species using NRLMSISE00."""
    f107a = 150.0
    f107 = 150.0
    ap = 8.0
    
    dens, temp = msise_model(
        datetime(year, 1, 1),
        alt_km,
        lat_deg,
        lon_deg,
        f107a,
        f107,
        ap
    )
    
    dens = np.array(dens, dtype=float)
    
    # dens[0]=Total mass, [1]=O, [2]=N2, [3]=O2, [7]=N (in cm^-3)
    species_map = {
        'O': 1,
        'N2': 2,
        'O2': 3,
        'N': 7
    }
    
    if species == 'Moyenne':
        # Return average of main species
        return (dens[1] + dens[2] + dens[3] + dens[7]) / 4.0
    else:
        idx = species_map.get(species, 2)
        return float(dens[idx])


def build_density_sphere(alt_km, lat_center_deg, lon_center_deg, species='N2', n_lat=40, n_lon=80, year=2020):
    """Build a sphere with radius modulated by atmospheric density of selected species."""
    
    # Create latitude/longitude grids (relative to center point)
    lats = np.linspace(-90, 90, n_lat)
    lons = np.linspace(-180, 180, n_lon)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    
    # Add center offset
    lat_grid = lat_grid + lat_center_deg
    lon_grid = lon_grid + lon_center_deg
    
    # Clamp latitude to [-90, 90]
    lat_grid = np.clip(lat_grid, -90, 90)
    
    # Get densities at all points
    density_grid = np.zeros_like(lat_grid, dtype=float)
    
    for i in range(n_lat):
        for j in range(n_lon):
            lat = lat_grid[i, j]
            lon = lon_grid[i, j]
            density_grid[i, j] = get_species_density(alt_km, lat, lon, species=species, year=year)
    
    # Normalize density to create radius variations
    # Use log scale to enhance contrast
    density_log = np.log10(density_grid + 1e-30)
    density_min = np.percentile(density_log, 5)
    density_max = np.percentile(density_log, 95)
    
    # Normalize to [0, 1] with clipping
    density_normalized = np.clip((density_log - density_min) / (density_max - density_min), 0, 1)
    
    # Create radius variations: base radius + modulation
    base_radius = alt_km
    radius_variation = 150 * (density_normalized ** 1.8)  # Stronger exponent for more contrast
    radius_grid = base_radius + radius_variation
    
    # Convert to Cartesian coordinates
    x_grid = np.zeros_like(radius_grid)
    y_grid = np.zeros_like(radius_grid)
    z_grid = np.zeros_like(radius_grid)
    
    for i in range(n_lat):
        for j in range(n_lon):
            x, y, z = spherical_to_cartesian(radius_grid[i, j], lat_grid[i, j], lon_grid[i, j])
            x_grid[i, j] = x
            y_grid[i, j] = y
            z_grid[i, j] = z
    
    return x_grid, y_grid, z_grid, density_normalized


def create_figure():
    """Create interactive figure with altitude slider and species buttons."""
    print("Creating interactive atmosphere sphere...")
    
    # Initial parameters
    initial_alt = 200
    initial_species = 'N2'
    lat_center = 0
    lon_center = 0
    year = 2020
    
    # Altitude range
    alt_values = list(range(100, 501, 25))  # 100 to 500 km with 25 km steps
    
    # Compute all density spheres for all altitudes and species
    print(f"  Computing density grids (this may take a minute)...")
    data_cache = {}
    
    species_list = ['N2', 'O2', 'N', 'O', 'Moyenne']
    
    for species in species_list:
        data_cache[species] = {}
        for alt in alt_values:
            x, y, z, dens = build_density_sphere(alt, lat_center, lon_center, species=species, year=year)
            data_cache[species][alt] = (x, y, z, dens)
            print(f"    {species:6s} @ {alt:3d} km ✓")
    
    # Create initial figure
    x, y, z, dens = data_cache[initial_species][initial_alt]
    
    # Species colors
    species_colors = {
        'N2': 'Viridis',
        'O2': 'Plasma',
        'N': 'Inferno',
        'O': 'Magma',
        'Moyenne': 'Blues'
    }
    
    fig = go.Figure(data=[
        go.Surface(
            x=x, y=y, z=z,
            surfacecolor=dens,
            colorscale=species_colors[initial_species],
            colorbar=dict(title=f'Densité {initial_species}<br>(normalisée)'),
            name=initial_species
        )
    ])
    
    # Create frames for slider
    frames = []
    
    for alt in alt_values:
        x, y, z, dens = data_cache[initial_species][alt]
        frames.append(
            go.Frame(
                data=[
                    go.Surface(
                        x=x, y=y, z=z,
                        surfacecolor=dens,
                        colorscale=species_colors[initial_species],
                        colorbar=dict(title=f'Densité {initial_species}<br>(normalisée)'),
                        name=initial_species
                    )
                ],
                name=str(alt),
                layout=go.Layout(
                    title=f'Densité atmosphérique: {initial_species}<br>Altitude: {alt} km | Latitude: {lat_center}° | Longitude: {lon_center}°'
                )
            )
        )
    
    # Create buttons for species selection
    buttons_species = []
    for species in species_list:
        # Create frames for this species and add to main frames list
        for alt in alt_values:
            x, y, z, dens = data_cache[species][alt]
            frames.append(
                go.Frame(
                    data=[
                        go.Surface(
                            x=x, y=y, z=z,
                            surfacecolor=dens,
                            colorscale=species_colors[species],
                            colorbar=dict(title=f'Densité {species}<br>(normalisée)'),
                            name=species
                        )
                    ],
                    name=f'{species}_{alt}',
                    layout=go.Layout(
                        title=f'Densité atmosphérique: {species}<br>Altitude: {alt} km | Latitude: {lat_center}° | Longitude: {lon_center}°'
                    )
                )
            )
        
        button = dict(
            label=species,
            method='animate',
            args=[
                [str(initial_alt)],
                {
                    'frame': {'duration': 500, 'redraw': True},
                    'fromcurrent': True,
                    'mode': 'immediate',
                    'transition': {'duration': 300}
                }
            ]
        )
        buttons_species.append(button)
    
    fig.frames = frames
    
    # Update layout with slider and buttons
    fig.update_layout(
        title=f'Densité atmosphérique: {initial_species}<br>Altitude: {initial_alt} km | Latitude: {lat_center}° | Longitude: {lon_center}°',
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data',
            camera=dict(eye=dict(x=2, y=2, z=1.5))
        ),
        width=1200,
        height=900,
        updatemenus=[
            dict(
                type='buttons',
                direction='down',
                x=0.0,
                y=1.0,
                buttons=buttons_species,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='gray',
                borderwidth=1
            )
        ],
        sliders=[
            {
                'active': alt_values.index(initial_alt),
                'yanchor': 'top',
                'y': -0.05,
                'xanchor': 'left',
                'x': 0.15,
                'currentvalue': {
                    'prefix': 'Altitude: ',
                    'suffix': ' km',
                    'visible': True,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': 'black'}
                },
                'transition': {'duration': 300},
                'pad': {'b': 10, 't': 50},
                'len': 0.6,
                'steps': [
                    {
                        'args': [[str(alt)], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                        'method': 'animate',
                        'label': str(alt)
                    }
                    for alt in alt_values
                ]
            }
        ]
    )
    
    # Save figure
    output_file = 'figures/E09/atmosphere/atmosphere_sphere_interactive.html'
    fig.write_html(output_file)
    print(f"Figure interactive sauvegardée: {output_file}")


if __name__ == '__main__':
    create_figure()
