# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Horizontal cell metrics (``dx``, ``dy``) carried from the source grid.

``areacello`` says how much area a cell covers. It does not say how wide or how
tall the cell is, and cross-section diagnostics need exactly that: transport
through a section sums ``velocity * dx * dz`` or ``velocity * dy * dz`` along
the section, so AMOC and every other transport number depends on the true cell
widths.

Analysis code used to recover them downstream by differencing the cell
coordinates. On a rectilinear grid that is exact, up to the radius the sphere is
given. On a curvilinear grid it is not. Once rows of the grid stop following
lines of constant latitude, the coordinate difference between neighbours stops
describing the cell: measured against OM4's own ``dxt``/``dyt``, the estimate
holds to 0.3% south of 60N, falls to 82% between 70N and 80N, and to 71% north
of 80N, with the worst cells off by a factor of seven.

MOM6 publishes the real metrics on the native grid, so we carry those rather
than derive them. The regridded output lands on a rectilinear target grid, where
deriving from the cell bounds is correct, and where there is nothing to carry.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

# Matches the radius `horizontal_regrid` uses to recompute `areacello` for the
# target grid, so the metrics and the area on a given store describe the same
# sphere. It is the polar radius rather than the mean one, which makes both
# about 0.24% small; that is a pre-existing choice and changing it would move
# every published area, so it is left alone here and tracked separately.
EARTH_RADIUS_M = 6356e3

# The native metrics we carry, and the MOM6 fields they come from. `dxt`/`dyt`
# are "Delta(x)/Delta(y) at thickness/tracer points", already in meters and
# already on the tracer grid that `areacello`, `geolat` and `geolon` use, so no
# interpolation is involved.
NATIVE_METRIC_SOURCES: dict[str, str] = {"dx": "dxt", "dy": "dyt"}

METRIC_ATTRS: dict[str, dict[str, str]] = {
    "dx": {
        "long_name": "Cell width in the i direction at tracer points",
        "units": "m",
    },
    "dy": {
        "long_name": "Cell height in the j direction at tracer points",
        "units": "m",
    },
}


def native_grid_metrics(ds_grid: xr.Dataset) -> dict[str, xr.DataArray]:
    """Read ``dx``/``dy`` off a model's own static grid file.

    Args:
        ds_grid: A MOM6 ``ocean_static`` dataset, which carries the horizontal
            metrics alongside `areacello` and the 2-D cell centers.

    Returns:
        A mapping of ``dx``/``dy`` to the native fields, ready to assign as
        coordinates.

    Raises:
        ValueError: If the grid file carries no metrics, or carries them on a
            different grid than `areacello`. Deriving them instead is only valid
            on a rectilinear grid, so there is nothing safe to fall back to.
    """
    missing = [src for src in NATIVE_METRIC_SOURCES.values() if src not in ds_grid]
    if missing:
        raise ValueError(
            f"Native grid file carries no {missing}. Cell widths cannot be "
            "derived from the coordinates on a curvilinear grid, where the "
            "spacing between neighbouring cell centers stops describing the "
            "cell once the grid folds. Point at a grid file that carries the "
            "model's own metrics."
        )

    metrics = {}
    for name, source in NATIVE_METRIC_SOURCES.items():
        metric = ds_grid[source]
        if "areacello" in ds_grid and metric.dims != ds_grid["areacello"].dims:
            raise ValueError(
                f"Native metric {source!r} has dimensions {metric.dims}, but "
                f"'areacello' has {ds_grid['areacello'].dims}. The metrics must "
                "be given at tracer points, on the same grid as the area."
            )
        metrics[name] = metric.astype("float64").assign_attrs(METRIC_ATTRS[name])
    return metrics


def rectilinear_grid_metrics(
    ds: xr.Dataset, *, radius_m: float = EARTH_RADIUS_M
) -> dict[str, xr.DataArray]:
    """Derive ``dx``/``dy`` from a rectilinear grid's cell bounds.

    ``dy`` is the meridional arc between the cell's south and north edges, and
    ``dx`` the zonal arc between its west and east edges at the cell's own
    latitude:

        dy = R * (lat_north - lat_south)
        dx = R * (lon_east - lon_west) * cos(lat_center)

    Args:
        ds: A grid carrying 2-D ``lat``/``lon`` cell centers on ``(y, x)`` and
            2-D ``lat_b``/``lon_b`` cell bounds on ``(y_b, x_b)``.
        radius_m: Radius of the sphere, in meters.

    Returns:
        A mapping of ``dx``/``dy`` to arrays on ``(y, x)``.

    Raises:
        ValueError: If the bounds are not separable, which means the grid is
            curvilinear and this derivation does not hold on it.
    """
    _require_separable_bounds(ds)

    lat = np.deg2rad(np.asarray(ds["lat"].values, dtype=np.float64))
    lat_b = np.deg2rad(np.asarray(ds["lat_b"].values, dtype=np.float64))
    lon_b = np.deg2rad(np.asarray(ds["lon_b"].values, dtype=np.float64))

    # Bounds are one wider than centers in each direction, so differencing along
    # a dimension lands back on the centers.
    dy = radius_m * (lat_b[1:, :-1] - lat_b[:-1, :-1])
    dx = radius_m * (lon_b[:-1, 1:] - lon_b[:-1, :-1]) * np.cos(lat)

    dims = ds["lat"].dims
    return {
        "dx": xr.DataArray(dx, dims=dims, attrs=METRIC_ATTRS["dx"]),
        "dy": xr.DataArray(dy, dims=dims, attrs=METRIC_ATTRS["dy"]),
    }


def _require_separable_bounds(ds: xr.Dataset, *, tolerance_deg: float = 1e-9) -> None:
    """Refuse to derive metrics from bounds that vary along both dimensions.

    On a rectilinear grid every row of `lat_b` is the same and every column of
    `lon_b` is the same, which is what makes a single difference describe the
    whole row or column. A curvilinear grid breaks that, and the derivation
    below would silently return the wrong widths rather than fail.
    """
    lat_b = np.asarray(ds["lat_b"].values, dtype=np.float64)
    lon_b = np.asarray(ds["lon_b"].values, dtype=np.float64)

    lat_spread = float(np.nanmax(np.abs(lat_b - lat_b[:, :1])))
    lon_spread = float(np.nanmax(np.abs(lon_b - lon_b[:1, :])))
    if lat_spread > tolerance_deg or lon_spread > tolerance_deg:
        raise ValueError(
            "Cannot derive cell metrics from these bounds: 'lat_b' varies by "
            f"{lat_spread:g} degrees along x and 'lon_b' by {lon_spread:g} "
            "degrees along y, so the grid is curvilinear. Carry the model's own "
            "'dx'/'dy' instead of deriving them."
        )
