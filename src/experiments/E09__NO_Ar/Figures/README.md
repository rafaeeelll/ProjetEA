# E09 Figures Scripts

Scripts (grouped by report section):

- `fig_a_intake_physics.py` -> figures 1, 2, 3
- `fig_b_feasibility.py` -> figures 4, 5, 6
- `fig_c_plasma_0d.py` -> figures 7, 8, 9, 10
- `fig_d_propulsive_performance.py` -> figures 11, 12, 13
- `fig_e_orbital_variability.py` -> figures 14, 15, 16
- `fig_f_optimization.py` -> figures 17, 18

Output directory:

- `PROJECT_ROOT/figures/E09/`

Example usage (from project root):

```bash
.venv/bin/python src/experiments/E09__NO_Ar/Figures/fig_a_intake_physics.py
.venv/bin/python src/experiments/E09__NO_Ar/Figures/fig_b_feasibility.py --fast --n-alt 9 --n-area 9
.venv/bin/python src/experiments/E09__NO_Ar/Figures/fig_c_plasma_0d.py --fast
.venv/bin/python src/experiments/E09__NO_Ar/Figures/fig_d_propulsive_performance.py --fast
.venv/bin/python src/experiments/E09__NO_Ar/Figures/fig_e_orbital_variability.py --fast
.venv/bin/python src/experiments/E09__NO_Ar/Figures/fig_f_optimization.py
```

