from datetime import datetime


def main():
    try:
        from nrlmsise00 import msise_model  # type: ignore
    except Exception as exc:
        print(f"Import error: {exc}")
        return

    altitude_km = 250.0
    lat = 0.0
    lon = 0.0
    date = datetime(2020, 1, 1, 12, 0, 0)
    f107 = 150.0
    f107a = 150.0
    ap = 4

    dens, temp = msise_model(date, altitude_km, lat, lon, f107a, f107, ap)

    print("MSIS output (raw):")
    print("densities:", dens)
    print("temperatures:", temp)

    # Typical indices used in many wrappers:
    # d[1]=O, d[2]=N2, d[3]=O2, d[7]=N (in cm^-3 in some implementations)
    try:
        print(f"O  (d[1]): {dens[1]} cm^-3")
        print(f"N2 (d[2]): {dens[2]} cm^-3")
        print(f"O2 (d[3]): {dens[3]} cm^-3")
        print(f"N  (d[7]): {dens[7]} cm^-3")
    except Exception:
        pass


if __name__ == "__main__":
    main()
