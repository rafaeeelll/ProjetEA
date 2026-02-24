#!/usr/bin/env python3
"""Visualisation 3D interactive (Plotly): demi-sphère autour de la Terre représentant
la densité moyenne (somme des espèces) en utilisant le module `nrlmsise00`.

Fonctions:
- chargement des densités (N, N2, O, O2) via nrlmsise00.msise_model
- tracé d'une surface sphérique 3D interactive colorée par densité
- vous pouvez tourner, zoomer avec la souris

Usage exemple:
  python species_density_sphere.py --hemisphere north --out figures/species_density_sphere.html --show
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Callable

import numpy as np
import plotly.graph_objects as go


def get_density_callable(date: datetime = None) -> Callable:
    """Build a callable that uses nrlmsise00 to get total species density.

    Parameters:
    - date: datetime object (default: 2020-01-01 12:00:00)

    Returns:
    - function(lat_array, lon_array, alt_array_km) -> density_array (m^-3)
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

    def get_density(lat, lon, alt_km):
        """
        Parameters:
        - lat: scalar or array (degrees)
        - lon: scalar or array (degrees)
        - alt_km: scalar or array (km above surface)

        Returns:
        - density: scalar or array (m^-3) — sum of N, N2, O, O2
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

        densities = np.zeros_like(lat_flat, dtype=float)

        for i in range(len(lat_flat)):
            try:
                dens, temp = msise_model(date, alt_flat[i], lat_flat[i], lon_flat[i], f107a, f107, ap)
                dens = np.array(dens, dtype=float)
                # NRLMSISE outputs number densities in cm^-3; convert to m^-3
                # dens[1]=O, dens[2]=N2, dens[3]=O2, dens[7]=N (from nrlmsise documentation)
                n_N2 = dens[2] * 1e6
                n_O2 = dens[3] * 1e6
                n_O = dens[1] * 1e6
                n_N = dens[7] * 1e6
                densities[i] = n_N2 + n_O2 + n_O + n_N
            except Exception as e:
                print(f'Warning: msise_model failed at lat={lat_flat[i]}, lon={lon_flat[i]}, alt={alt_flat[i]}: {e}', file=sys.stderr)
                densities[i] = np.nan

        return densities.reshape(out_shape)

    return get_density


def sph_coords_to_cart(radius, lat_deg, lon_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return x, y, z


def build_density_grid(get_density, radius_km: float, n_lat: int, n_lon: int) -> tuple:
    """Return (X, Y, Z, density) grids on sphere of given radius (km).

    - get_density(lat_array, lon_array, alt_or_radius_array) -> array
    - We pass alt in km (radius_km - R_earth_km)
    """
    # Earth's mean radius in km
    R_earth = 6371.0
    lat_edges = np.linspace(-90.0, 90.0, n_lat)
    lon_edges = np.linspace(0.0, 360.0, n_lon, endpoint=False)
    lon2d, lat2d = np.meshgrid(lon_edges, lat_edges)

    # call density function with lat, lon, alt (alt in km above surface)
    alt_km = radius_km - R_earth
    lat_flat = lat2d.ravel()
    lon_flat = lon2d.ravel()
    alt_flat = np.full_like(lat_flat, alt_km)

    dens_flat = get_density(lat_flat, lon_flat, alt_flat)
    dens = np.asarray(dens_flat).reshape(lat2d.shape)

    x, y, z = sph_coords_to_cart(radius_km, lat2d, lon2d)
    return x, y, z, dens


def build_hemisphere_grid(get_density, radius_km: float, n_lat: int, n_lon: int, 
                          lat_range: tuple = (0, 90), lon_range: tuple = (0, 180)) -> tuple:
    """Return (X, Y, Z, density) grids on hemisphere of given radius (km).

    Parameters:
    - lat_range: tuple (lat_min, lat_max) in degrees
    - lon_range: tuple (lon_min, lon_max) in degrees
    """
    R_earth = 6371.0
    lat_edges = np.linspace(lat_range[0], lat_range[1], n_lat)
    lon_edges = np.linspace(lon_range[0], lon_range[1], n_lon)
    lon2d, lat2d = np.meshgrid(lon_edges, lat_edges)

    alt_km = radius_km - R_earth
    lat_flat = lat2d.ravel()
    lon_flat = lon2d.ravel()
    alt_flat = np.full_like(lat_flat, alt_km)

    dens_flat = get_density(lat_flat, lon_flat, alt_flat)
    dens = np.asarray(dens_flat).reshape(lat2d.shape)

    x, y, z = sph_coords_to_cart(radius_km, lat2d, lon2d)
    return x, y, z, dens


def plot_sphere_surface(fig, X, Y, Z, dens, name='Densité'):
    """Add a surface trace to Plotly figure."""
    # Normalize density for colorscale
    dens_min = np.nanmin(dens)
    dens_max = np.nanmax(dens)
    
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=dens,
        colorscale='Plasma',
        showscale=True,
        colorbar=dict(title='Densité (m⁻³)', thickness=20, len=0.7),
        name=name,
        hovertemplate='X: %{x:.0f}<br>Y: %{y:.0f}<br>Z: %{z:.0f}<br>Densité: %{surfacecolor:.2e}<extra></extra>'
    ))


def overlay_isodensity(fig, X, Y, Z, dens, levels=8):
    """Overlay contour lines on the surface (optional)."""
    # For now, just use surface color to show contours
    pass  # Plotly surface colorscale already shows density variations well


def parse_args():
    p = argparse.ArgumentParser(description='Trace interactive (Plotly) la densité moyenne sur une demi-sphère autour de la Terre')
    p.add_argument('--date', type=str, default='2020-01-01', help='Date au format YYYY-MM-DD (par défaut 2020-01-01)')
    p.add_argument('--radius-offset', type=float, default=200.0, help='Altitude (km) au-dessus de la surface (défaut 200 km)')
    p.add_argument('--n-lat', type=int, default=48, help='Nombre de points en latitude (défaut 48)')
    p.add_argument('--n-lon', type=int, default=96, help='Nombre de points en longitude (défaut 96)')
    p.add_argument('--hemisphere', type=str, choices=['north', 'south', 'east', 'west', 'full'], default='north', help='Région à tracer')
    p.add_argument('--cmap', type=str, default='Plasma', help='Colormap Plotly (ex: Plasma, Viridis, RdBu, Jet)')
    p.add_argument('--out', type=str, default=os.path.join('figures', 'species_density_sphere.html'), help='Fichier HTML interactif (défaut .html)')
    p.add_argument('--show', action='store_true', help='Ouvrir dans le navigateur après génération')
    return p.parse_args()


def main():
    args = parse_args()

    # parse date
    try:
        date = datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        print(f'Erreur: format de date invalide. Utilisez YYYY-MM-DD', file=sys.stderr)
        raise

    # get density callable
    try:
        get_density = get_density_callable(date)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        raise

    # prepare shells (for now, just one shell)
    R_earth = 6371.0
    base_radius = R_earth + args.radius_offset

    # determine lat/lon ranges based on hemisphere choice
    hemisphere_ranges = {
        'north': ((0, 90), (0, 180)),
        'south': ((-90, 0), (0, 180)),
        'east': ((-90, 90), (0, 180)),
        'west': ((-90, 90), (180, 360)),
        'full': ((-90, 90), (0, 360)),
    }
    lat_range, lon_range = hemisphere_ranges[args.hemisphere]

    # compute density grid
    print(f'Calcul de la densité sur la coquille r={base_radius:.1f} km ...')
    if args.hemisphere == 'full':
        X, Y, Z, dens = build_density_grid(get_density, base_radius, args.n_lat, args.n_lon)
    else:
        X, Y, Z, dens = build_hemisphere_grid(get_density, base_radius, args.n_lat, args.n_lon, lat_range, lon_range)

    # Create Plotly figure
    fig = go.Figure()
    
    # Add surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=dens,
        colorscale=args.cmap,
        showscale=True,
        colorbar=dict(
            title='Densité (m⁻³)',
            thickness=20,
            len=0.7,
            x=1.02
        ),
        hovertemplate='X: %{x:.0f} km<br>Y: %{y:.0f} km<br>Z: %{z:.0f} km<br>Densité: %{surfacecolor:.2e} m⁻³<extra></extra>'
    ))

    # Update layout for better viewing
    fig.update_layout(
        title=f'Densité atmosphérique à {args.radius_offset} km ({args.hemisphere.upper()})<br>nrlmsise00 | {args.date}',
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=1200,
        height=800,
        margin=dict(l=0, r=0, b=0, t=50),
        hovermode='closest'
    )

    # Save to HTML
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    
    # Determine output format based on extension
    if args.out.endswith('.html'):
        fig.write_html(args.out)
        print(f'Figure interactive sauvegardée: {args.out}')
        if args.show:
            fig.show()
    else:
        # Fall back to static image (requires kaleido)
        try:
            fig.write_image(args.out, width=1200, height=800)
            print(f'Figure sauvegardée: {args.out}')
        except Exception as e:
            print(f'Warning: impossible de sauvegarder en {os.path.splitext(args.out)[1]}', file=sys.stderr)
            print('  (installez kaleido pour les images statiques: pip install kaleido)', file=sys.stderr)
            # Save as HTML instead
            html_out = os.path.splitext(args.out)[0] + '.html'
            fig.write_html(html_out)
            print(f'Fichier HTML sauvegardé à la place: {html_out}')
        if args.show:
            fig.show()


if __name__ == '__main__':
    main()
