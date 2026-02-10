"""
Atmosphere density sphere with interactive altitude, latitude, and longitude selection.
The sphere radius is modulated by atmospheric density at each point.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


def get_density_at_point(alt_km, lat_deg, lon_deg, year=2020):
    """Get total atmospheric density at a point using NRLMSISE00."""
    # Default space weather indices
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
    
    # dens[0] = total mass density (g/cm^3)
    return float(dens[0])


def build_density_sphere(alt_km, lat_center_deg, lon_center_deg, n_lat=50, n_lon=100, year=2020):
    """Build a sphere with radius modulated by atmospheric density."""
    
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
            density_grid[i, j] = get_density_at_point(alt_km, lat, lon, year)
    
    # Normalize density to create radius variations
    # Use log scale to enhance contrast
    density_log = np.log10(density_grid + 1e-30)
    density_min = np.percentile(density_log, 5)
    density_max = np.percentile(density_log, 95)
    
    # Normalize to [0, 1] with clipping
    density_normalized = np.clip((density_log - density_min) / (density_max - density_min), 0, 1)
    
    # Create radius variations: base radius + modulation
    base_radius = alt_km
    radius_variation = 100 * (density_normalized ** 1.5)  # Exponent for contrast
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


def create_figure_for_params(alt_km, lat_deg, lon_deg, year=2020):
    """Create figure for given parameters."""
    print(f"  Computing density sphere: alt={alt_km} km, lat={lat_deg}°, lon={lon_deg}°")
    
    x, y, z, density = build_density_sphere(alt_km, lat_deg, lon_deg, year=year)
    
    fig = go.Figure(data=[
        go.Surface(
            x=x, y=y, z=z,
            surfacecolor=density,
            colorscale='Viridis',
            colorbar=dict(title='Density\n(normalized)'),
            name='Atmosphere'
        )
    ])
    
    return fig


def main():
    """Build interactive atmosphere explorer with sliders."""
    print("Creating interactive atmosphere explorer...")
    
    # Initial parameters
    initial_alt = 200
    initial_lat = 0
    initial_lon = 0
    year = 2020
    
    # Create initial figure
    fig = create_figure_for_params(initial_alt, initial_lat, initial_lon, year=year)
    
    # Define slider ranges and steps
    alt_values = np.arange(100, 501, 50)  # 100 to 500 km
    lat_values = np.arange(-90, 91, 30)   # -90 to 90 deg
    lon_values = np.arange(-180, 180, 30) # -180 to 180 deg
    
    # Create all figures for different altitude/lat/lon combinations
    figures_dict = {}
    
    for alt in alt_values:
        for lat in lat_values:
            for lon in lon_values:
                key = (alt, lat, lon)
                x, y, z, density = build_density_sphere(alt, lat, lon, year=year)
                figures_dict[key] = (x, y, z, density)
    
    # Update layout with sliders
    fig.update_layout(
        title=f'Sphère de densité atmosphérique<br>Altitude: {initial_alt} km | Latitude: {initial_lat}° | Longitude: {initial_lon}°',
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data',
            camera=dict(eye=dict(x=2, y=2, z=1.5))
        ),
        width=1200,
        height=900,
        sliders=[
            {
                'active': 0,
                'yanchor': 'top',
                'y': 0,
                'xanchor': 'left',
                'x': 0,
                'currentvalue': {
                    'prefix': 'Altitude: ',
                    'suffix': ' km',
                    'visible': True,
                    'xanchor': 'center',
                    'font': {'size': 14}
                },
                'transition': {'duration': 300},
                'pad': {'b': 10, 't': 50},
                'len': 0.3,
                'steps': [
                    {
                        'args': [[alt], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                        'method': 'animate',
                        'label': str(alt)
                    }
                    for alt in alt_values
                ]
            },
            {
                'active': 3,
                'yanchor': 'top',
                'y': -0.05,
                'xanchor': 'left',
                'x': 0,
                'currentvalue': {
                    'prefix': 'Latitude: ',
                    'suffix': '°',
                    'visible': True,
                    'xanchor': 'center',
                    'font': {'size': 14}
                },
                'transition': {'duration': 300},
                'pad': {'b': 10, 't': 10},
                'len': 0.3,
                'steps': [
                    {
                        'args': [[lat], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                        'method': 'animate',
                        'label': str(lat)
                    }
                    for lat in lat_values
                ]
            },
            {
                'active': 3,
                'yanchor': 'top',
                'y': -0.1,
                'xanchor': 'left',
                'x': 0,
                'currentvalue': {
                    'prefix': 'Longitude: ',
                    'suffix': '°',
                    'visible': True,
                    'xanchor': 'center',
                    'font': {'size': 14}
                },
                'transition': {'duration': 300},
                'pad': {'b': 10, 't': 10},
                'len': 0.3,
                'steps': [
                    {
                        'args': [[lon], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                        'method': 'animate',
                        'label': str(lon)
                    }
                    for lon in lon_values
                ]
            }
        ]
    )
    
    # Create frames for animation
    frames = []
    
    for alt in alt_values:
        for lat in lat_values:
            for lon in lon_values:
                key = (alt, lat, lon)
                x, y, z, density = figures_dict[key]
                
                frames.append(
                    go.Frame(
                        data=[
                            go.Surface(
                                x=x, y=y, z=z,
                                surfacecolor=density,
                                colorscale='Viridis',
                                colorbar=dict(title='Density\n(normalized)'),
                                name='Atmosphere'
                            )
                        ],
                        name=f'alt={alt}_lat={lat}_lon={lon}',
                        layout=go.Layout(
                            title=f'Sphère de densité atmosphérique<br>Altitude: {alt} km | Latitude: {lat}° | Longitude: {lon}°'
                        )
                    )
                )
    
    fig.frames = frames
    
    # Save figure
    output_file = 'figures/atmosphere_sphere_interactive.html'
    fig.write_html(output_file)
    print(f"Figure interactive sauvegardée: {output_file}")


if __name__ == '__main__':
    main()
