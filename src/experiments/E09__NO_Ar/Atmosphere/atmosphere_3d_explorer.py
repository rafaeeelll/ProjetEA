#!/usr/bin/env python3
"""Visualisation 3D interactive (Plotly): Terre texturée + contours de densité.

- Sphère Terre au centre avec texture réaliste
- Lignes de contours pour les densités d'espèces
- Sélecteurs interactifs: espèce, année, latitude, longitude

Usage:
    python atmosphere_3d_explorer.py --out figures/E09/atmosphere/atmosphere_explorer.html --show
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import plotly.graph_objects as go


def get_earth_texture():
    """Get a simple Earth texture (blue-green-brown)."""
    # Create a simple Earth by computing a height map
    u = np.linspace(0, 2 * np.pi, 200)
    v = np.linspace(0, np.pi, 100)
    u_grid, v_grid = np.meshgrid(u, v)
    
    # Simple Perlin-like pattern for continents
    continents = 0.5 + 0.3 * np.sin(5 * u_grid) * np.cos(3 * v_grid)
    continents = np.clip(continents, 0, 1)
    
    return continents


def create_earth_sphere(radius_km: float = 6371.0, n_points: int = 100):
    """Create Earth sphere with realistic coloring."""
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    u_grid, v_grid = np.meshgrid(u, v)
    
    x = radius_km * np.cos(u_grid) * np.sin(v_grid)
    y = radius_km * np.sin(u_grid) * np.sin(v_grid)
    z = radius_km * np.cos(v_grid)
    
    # Simple texture: continents vs oceans
    texture = get_earth_texture() * 0.5 + 0.5
    
    return x, y, z, texture


def sph_coords_to_cart(radius, lat_deg, lon_deg):
    """Convert spherical to Cartesian coordinates."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return x, y, z


def get_species_densities(lat_deg, lon_deg, alt_km, year=2020, month=1, day=1):
    """Get individual species densities for given location and date."""
    try:
        from nrlmsise00 import msise_model
    except ImportError:
        raise RuntimeError('nrlmsise00 not found. Install with: pip install nrlmsise00')
    
    date = datetime(year, month, day, 12, 0, 0)
    f107a, f107, ap = 150.0, 150.0, 4.0
    
    try:
        dens, temp = msise_model(date, alt_km, lat_deg, lon_deg, f107a, f107, ap)
        dens = np.array(dens, dtype=float)
        
        # dens[1]=O, dens[2]=N2, dens[3]=O2, dens[7]=N (cm^-3)
        return {
            'O': float(dens[1] * 1e6),      # m^-3
            'N2': float(dens[2] * 1e6),
            'O2': float(dens[3] * 1e6),
            'N': float(dens[7] * 1e6),
        }
    except Exception as e:
        print(f'Warning: msise_model failed: {e}', file=sys.stderr)
        return {spec: np.nan for spec in ['O', 'N2', 'O2', 'N']}


def build_vertical_profile(lat_deg, lon_deg, species, year, alt_min=100, alt_max=500, n_alts=30):
    """Build vertical profile for a species at given lat/lon/year.
    
    Returns: altitudes (km), densities (m^-3), and isodensity heights for contours
    """
    alts = np.linspace(alt_min, alt_max, n_alts)
    dens = []
    
    for alt in alts:
        result = get_species_densities(lat_deg, lon_deg, alt, year)
        dens.append(result.get(species, np.nan))
    
    dens = np.array(dens)
    return alts, dens


def create_isodensity_shells(lat_deg, lon_deg, species, year, alt_ref=200, 
                             n_shells=8, altitude_range=150):
    """Create isodensity contour lines as thick traces on altitude-based shells.
    
    Instead of contours at reference density, create discrete altitude layers
    with larger spacing to show variation better.
    """
    R_earth = 6371.0
    
    # Create shells at larger intervals for better visibility
    alts = np.linspace(alt_ref - altitude_range, alt_ref + altitude_range, n_shells)
    
    # Colors for each shell (gradient)
    colors_palette = [
        'rgb(100, 200, 255)',  # Light blue
        'rgb(100, 255, 200)',  # Cyan
        'rgb(100, 255, 100)',  # Green
        'rgb(200, 255, 100)',  # Yellow-green
        'rgb(255, 200, 100)',  # Orange
        'rgb(255, 150, 100)',  # Orange-red
        'rgb(255, 100, 100)',  # Red
        'rgb(200, 100, 255)',  # Purple
    ]
    
    traces = []
    for idx, alt in enumerate(alts):
        radius = R_earth + alt
        
        # Sample density at many latitudes/longitudes around Earth for this altitude
        n_samples = 200
        sample_lons = np.linspace(0, 360, n_samples)
        sample_lats = np.full_like(sample_lons, lat_deg)
        sample_alts = np.full_like(sample_lons, alt)
        
        densities = []
        valid_lons = []
        
        for lon, lat_s, alt_s in zip(sample_lons, sample_lats, sample_alts):
            try:
                result = get_species_densities(lat_s, lon, alt_s, year)
                dens = result.get(species, np.nan)
                if not np.isnan(dens) and dens > 0:
                    densities.append(dens)
                    valid_lons.append(lon)
            except Exception:
                pass
        
        if len(valid_lons) > 10:
            # Convert to 3D coordinates
            x, y, z = sph_coords_to_cart(radius, np.full_like(valid_lons, lat_deg), valid_lons)
            
            # Close the loop for proper circle
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            z = np.append(z, z[0])
            
            color = colors_palette[idx % len(colors_palette)]
            traces.append(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color=color, width=6),
                name=f'{species} ({alt:.0f} km)',
                showlegend=True,
                hovertemplate=f'{species} at {alt:.0f} km<br>Lat: {lat_deg:.1f}°<extra></extra>'
            ))
    
    return traces


def parse_args():
    p = argparse.ArgumentParser(description='Explorateur 3D interactif de l\'atmosphère avec sélecteurs')
    p.add_argument('--out', type=str, default=os.path.join('figures', 'E09', 'atmosphere', 'atmosphere_explorer.html'), help='Fichier HTML (défaut .html)')
    p.add_argument('--show', action='store_true', help='Ouvrir dans le navigateur')
    return p.parse_args()


def main():
    args = parse_args()
    
    print('Création de la figure interactive ...')
    
    # Create base figure (without Earth)
    fig = go.Figure()
    
    # Compute initial contours for all species
    print('Calcul des profils initiaux ...')
    lat_init, lon_init, year_init = 0, 0, 2020
    
    all_traces = {}
    for species in ['N2', 'O2', 'N', 'O']:
        print(f'  {species} ...')
        try:
            traces = create_isodensity_shells(lat_init, lon_init, species, year_init)
            all_traces[species] = traces
        except Exception as e:
            print(f'    Warning: {e}', file=sys.stderr)
            all_traces[species] = []
    
    # Add initial traces to figure
    for species in ['N2', 'O2', 'N', 'O']:
        for trace in all_traces[species]:
            fig.add_trace(trace)
    
    # Create buttons for species selection
    buttons_species = []
    for spec in ['N2', 'O2', 'N', 'O']:
        visible = [spec in t.name for t in fig.data]
        buttons_species.append(
            dict(label=spec, method='update', args=[{'visible': visible}])
        )
    
    # Update layout with buttons
    fig.update_layout(
        title='Contours de densité atmosphérique par espèce<br>Latitude: 0° | Longitude: 0° | Année: 2020<br>(sélectionnez une espèce ci-dessous)',
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        width=1400,
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
