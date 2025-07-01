import numpy as np
import xarray as xr
import glob
import xesmf as xe
import os

region_name = "Kuro_new_2"

def fill_nan_diffusion_soft(arr, max_iter=1000, tol=1e-4):
    """Smooth NaNs near coastlines with Laplacian diffusion."""
    filled = arr.copy()
    nan_mask = np.isnan(filled)
    filled[nan_mask] = 0  # Initialize NaNs

    for iteration in range(max_iter):
        prev = filled.copy()

        laplace = 0.25 * (
            np.roll(filled, +1, axis=0) +
            np.roll(filled, -1, axis=0) +
            np.roll(filled, +1, axis=1) +
            np.roll(filled, -1, axis=1)
        )

        filled[nan_mask] = laplace[nan_mask]

        diff = np.nanmax(np.abs(filled - prev))
        if diff < tol:
            print(f"Diffusion converged after {iteration} iterations.")
            break

    return filled

def main():
    file_list = sorted(glob.glob(f"/data/Hannah/{region_name}/*.nc"))[:-1]

    print("Loading Dataset")
    ds = xr.open_mfdataset(
        file_list,
        combine="by_coords",
        engine="netcdf4",
        decode_timedelta=True,
        preprocess=lambda ds: ds[["SSH", "TLAT", "TLONG"]],
        chunks=None
    )
    print("Loaded!")

    # Get native POP grid shape
    n_j, n_i = ds["TLAT"].shape
    print(f"Native grid shape: (j={n_j}, i={n_i})")

    # Mask invalid lon/lat values
    mask = ((ds['TLONG'] != -1) & (ds['TLAT'] != -1)).compute()
    ds['TLONG'] = ds['TLONG'].where(mask, drop=False)
    ds['TLAT'] = ds['TLAT'].where(mask, drop=False)

    # Get domain bounds
    lon_min = float(np.floor(ds["TLONG"].where(ds["TLONG"] != -1).min()))
    lon_max = float(np.ceil(ds["TLONG"].where(ds["TLONG"] != -1).max()))
    lat_min = float(np.floor(ds["TLAT"].where(ds["TLAT"] != -1).min()))
    lat_max = float(np.ceil(ds["TLAT"].where(ds["TLAT"] != -1).max()))

    # Compute exact spacing to preserve shape
    dlon = (lon_max - lon_min) / (n_i - 1)
    dlat = (lat_max - lat_min) / (n_j - 1)

    # 1D and 2D arrays
    lon_1d = np.linspace(lon_min, lon_max, n_i)
    lat_1d = np.linspace(lat_min, lat_max, n_j)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    # Optional: bounds for conservative methods
    lon_b = np.linspace(lon_min - dlon / 2, lon_max + dlon / 2, n_i + 1)
    lat_b = np.linspace(lat_min - dlat / 2, lat_max + dlat / 2, n_j + 1)
    lon_b_2d, lat_b_2d = np.meshgrid(lon_b, lat_b)

    ds_out = xr.Dataset(
        {
            "lon": (["y", "x"], lon_2d),
            "lat": (["y", "x"], lat_2d),
            "lon_b": (["y_b", "x_b"], lon_b_2d),
            "lat_b": (["y_b", "x_b"], lat_b_2d),
        },
        coords={
            "x": np.arange(n_i),
            "y": np.arange(n_j),
            "x_b": np.arange(n_i + 1),
            "y_b": np.arange(n_j + 1),
        }
    )


    # Rename coordinates for xESMF
    ds_renamed = ds.rename({"TLONG": "lon", "TLAT": "lat"})

    print("Starting regridding")
    regridder = xe.Regridder(ds_renamed, ds_out, "bilinear", periodic=False, reuse_weights=False)
    dr_out = regridder(ds_renamed["SSH"])  # shape = (time, j, i), same as original
    print("Regridding done!")

    print("Starting Laplacian Diffusion")
    ssh_filled = np.empty_like(dr_out)

    for t in range(dr_out.sizes["time"]):
        ssh_slice = dr_out.isel(time=t).values
        land_mask = np.isnan(ssh_slice)
        print(f"Time step {t}: Filling {np.sum(land_mask)} NaNs")
        ssh_filled[t, :, :] = fill_nan_diffusion_soft(ssh_slice)

    # Construct new DataArray
    ssh_interp = xr.DataArray(
        ssh_filled,
        coords=dr_out.coords,
        dims=dr_out.dims,
        attrs=dr_out.attrs
    )

    # Save output
    outroot = os.path.expanduser("~/work/esn/CESM/Data")
    os.makedirs(outroot, exist_ok=True)

    np.save(f"{outroot}/ssh_{region_name}.npy", ssh_interp.values)
    np.save(f"{outroot}/lon_{region_name}.npy", ds_out["lon"].values)
    np.save(f"{outroot}/lat_{region_name}.npy", ds_out["lat"].values)

    print("Done! Regridded and diffused SSH, lon, and lat saved.")

if __name__ == "__main__":
    main()
