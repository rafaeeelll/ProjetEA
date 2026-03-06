#!/usr/bin/env python3
"""Visualisation 3D interactive (Plotly): Terre au centre + coquilles d'espèces.

Chaque espèce (N, N2, O, O2) a sa propre coquille sphérique colorée.
Le rayon de chaque coquille varie selon la densité locale:
- densité forte → rayon augmente (s'éloigne de la Terre)
- densité faible → rayon diminue (proche de la Terre)
La densité moyenne est à l'altitude donnée.

Usage exemple:
    python species_density_shells.py --altitude 200 --hemisphere north --out figures/E09/atmosphere/species_shells.html --show
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Callable

import numpy as np
import plotly.graph_objects as go


def get_species_densities_callable(date: datetime = None) -> Callable:
    """Build a callable that uses nrlmsise00 to get individual species densities.

    Parameters:
    - date: datetime object (default: 2020-01-01 12:00:00)

    Returns:
    - function(lat_array, lon_array, alt_array_km) -> dict of species->density_array (m^-3)
    """
    if date is None:
        date = datetime(2020, 1, 1, 12, 0, 0)

    try:
        from nrlmsise00 import msise_model  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            f'Impossible d\'importer nrlmsise00. Installez-le avec: pip install nrlmsise00\n{e}'
        )

    # Fixed space weather parameters
    f107a = 150.0
    f107 = 150.0
    ap = 4.0

    def get_species(lat, lon, alt_km):
        """
        Parameters:
        - lat: scalar or array (degrees)
        - lon: scalar or array (degrees)
        - alt_km: scalar or array (km above surface)

        Returns:
        - dict with species names as keys and density arrays (m^-3) as values
        """
        lat = np.atleast_1d(lat)
        lon = np.atleast_1d(lon)
        alt_km = np.atleast_1d(alt_km)

        # broadcast to same shape
        lat, lon, alt_km = np.broadcast_arrays(lat, lon, alt_km)
        out_shape = lat.shape
        lat_flat = lat.ravel()
        lon_flat = lon.ravel()
        alt_flat = alt_km.ravel()

        # NRLMSISE output indices:
        # dens[0] = He, dens[1] = O, dens[2] = N2, dens[3] = O2, dens[7] = N
        species_map = {
            'O': 1,
            'N2': 2,
            'O2': 3,
            'N': 7,
        }

        result = {spec: np.zeros_like(lat_flat, dtype=float) for spec in species_map.keys()}

        for i in range(len(lat_flat)):
            try:
                dens, temp = msise_model(date, alt_flat[i], lat_flat[i], lon_flat[i], f107a, f107, ap)
                dens = np.array(dens, dtype=float)
                # Convert cm^-3 to m^-3
                for spec, idx in species_map.items():
                    result[spec][i] = dens[idx] * 1e6
            except Exception as e:
                print(f'Warning: msise_model failed at lat={lat_flat[i]}, lon={lon_flat[i]}, alt={alt_flat[i]}: {e}', file=sys.stderr)
                for spec in species_map.keys():
                    result[spec][i] = np.nan

        # Reshape back to original shape
        for spec in result.keys():
            result[spec] = result[spec].reshape(out_shape)

        return result

    return get_species


def sph_coords_to_cart(radius, lat_deg, lon_deg):
    """Convert spherical to Cartesian coordinates."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return x, y, z


def build_species_shells(get_species, base_altitude_km: float, n_lat: int, n_lon: int,
                         lat_range: tuple = (0, 90), lon_range: tuple = (0, 180),
                         modulation_factor: float = 50.0) -> dict:
    """Build 3D shells for each species, with radius modulated by density.

    Parameters:
    - base_altitude_km: reference altitude in km
    - modulation_factor: how much the radius changes per unit density change
    - Returns: dict with species names and (X, Y, Z) coordinate arrays
    """
    R_earth = 6371.0
    base_radius = R_earth + base_altitude_km

    lat_edges = np.linspace(lat_range[0], lat_range[1], n_lat)
    lon_edges = np.linspace(lon_range[0], lon_range[1], n_lon)
    lon2d, lat2d = np.meshgrid(lon_edges, lat_edges)

    lat_flat = lat2d.ravel()
    lon_flat = lon2d.ravel()
    alt_flat = np.full_like(lat_flat, base_altitude_km)

    # Get densities for all species
    species_dens = get_species(lat_flat, lon_flat, alt_flat)

    result = {}
    for spec, dens_flat in species_dens.items():
        dens = dens_flat.reshape(lat2d.shape)
        
        # Normalize density to modulation: 
        # radius_offset = modulation_factor * log(density) if density > 0
        dens_safe = np.where(dens > 0, dens, np.nan)
        log_dens = np.log10(np.maximum(dens_safe, 1e-20))
        log_dens_norm = (log_dens - np.nanmin(log_dens)) / (np.nanmax(log_dens) - np.nanmin(log_dens) + 1e-10)
        
        # Radius for this species: base + modulation
        radius_2d = base_radius + modulation_factor * (log_dens_norm - 0.5)
        
        # Convert to Cartesian
        x, y, z = sph_coords_to_cart(radius_2d, lat2d, lon2d)
        
        result[spec] = {
            'X': x,
            'Y': y,
            'Z': z,
            'dens': dens,
            'radius': radius_2d,
        }

    return result


def add_earth_sphere(fig, radius_km: float = 6371.0, n_points: int = 50):
    """Add Earth sphere to the figure (simple textured sphere)."""
    # Create Earth as a simple sphere with blue-brown coloring
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Simple color: blue-green for oceans, brown for continents (random pattern)
    colors = np.random.choice([0, 1], size=x.shape)
    
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        surfacecolor=colors,
        colorscale=[[0, 'rgb(30, 100, 200)'], [1, 'rgb(139, 90, 43)']],
        showscale=False,
        name='Earth',
        hovertemplate='Earth<extra></extra>',
        opacity=0.9
    ))


def parse_args():
    p = argparse.ArgumentParser(description='Visualise les densités des espèces atmosphériques en coquilles sphériques interactives')
    p.add_argument('--date', type=str, default='2020-01-01', help='Date au format YYYY-MM-DD')
    p.add_argument('--altitude', type=float, default=200.0, help='Altitude (km) de référence (défaut 200 km)')
    p.add_argument('--n-lat', type=int, default=48, help='Nombre de points en latitude (défaut 48)')
    p.add_argument('--n-lon', type=int, default=96, help='Nombre de points en longitude (défaut 96)')
    p.add_argument('--hemisphere', type=str, choices=['north', 'south', 'east', 'west', 'full'], default='north', help='Région à tracer')
    p.add_argument('--modulation', type=float, default=50.0, help='Facteur de modulation du rayon (défaut 50)')
    p.add_argument('--out', type=str, default=os.path.join('figures', 'E09', 'atmosphere', 'species_shells.html'), help='Fichier HTML interactif')
    p.add_argument('--show', action='store_true', help='Ouvrir dans le navigateur')
    return p.parse_args()


def main():
    args = parse_args()

    # parse date
    try:
        date = datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        print(f'Erreur: format de date invalide. Utilisez YYYY-MM-DD', file=sys.stderr)
        raise

    # get species density callable
    try:
        get_species = get_species_densities_callable(date)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        raise

    # determine lat/lon ranges
    hemisphere_ranges = {
        'north': ((0, 90), (0, 180)),
        'south': ((-90, 0), (0, 180)),
        'east': ((-90, 90), (0, 180)),
        'west': ((-90, 90), (180, 360)),
        'full': ((-90, 90), (0, 360)),
    }
    lat_range, lon_range = hemisphere_ranges[args.hemisphere]

    print(f'Calcul des densités d\'espèces à {args.altitude} km ...')
    species_shells = build_species_shells(
        get_species, 
        args.altitude, 
        args.n_lat, 
        args.n_lon,
        lat_range, 
        lon_range,
        modulation_factor=args.modulation
    )

    # Create Plotly figure
    fig = go.Figure()

    # Add Earth at the center
    print('Ajout de la Terre ...')
    add_earth_sphere(fig, radius_km=6371.0, n_points=30)

    # Color map for species
    species_colors = {
        'N': 'rgb(255, 100, 100)',   # Red
        'O': 'rgb(100, 200, 255)',   # Blue
        'N2': 'rgb(100, 255, 100)',  # Green
        'O2': 'rgb(255, 200, 100)',  # Orange
    }

    # Add each species as a shell
    print('Ajout des coquilles d\'espèces ...')
    for spec in ['N2', 'O2', 'N', 'O']:  # Order: largest to smallest visually
        if spec not in species_shells:
            continue

        data = species_shells[spec]
        X, Y, Z = data['X'], data['Y'], data['Z']
        dens = data['dens']

        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=dens,
            colorscale='Viridis',
            showscale=(spec == 'N'),  # Only show colorbar for last species
            colorbar=dict(
                title=f'Densité<br>{spec}<br>(m⁻³)',
                thickness=15,
                len=0.5,
                x=1.05
            ),
            name=spec,
            opacity=0.7,
            hovertemplate=f'{spec}<br>Densité: %{{surfacecolor:.2e}} m⁻³<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        title=f'Densités des espèces atmosphériques à {args.altitude} km<br>({args.hemisphere.upper()}) | nrlmsise00 | {args.date}',
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=2.0, y=2.0, z=1.5)
            )
        ),
        width=1400,
        height=900,
        margin=dict(l=0, r=150, b=0, t=70),
        hovermode='closest'
    )

    # Save to HTML
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.write_html(args.out)
    print(f'Figure interactive sauvegardée: {args.out}')
    
    if args.show:
        fig.show()


if __name__ == '__main__':
    main()
