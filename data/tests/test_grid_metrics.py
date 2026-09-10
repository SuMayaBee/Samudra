# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Horizontal cell metrics carried from, or derived against, the source grid."""

import numpy as np
import pytest
import xarray as xr
from ocean_preprocessing.grid_metrics import (
    EARTH_RADIUS_M,
    native_grid_metrics,
    rectilinear_grid_metrics,
)

NY, NX = 3, 4


@pytest.fixture
def rectilinear():
    """A regular grid: 30 degrees of latitude by 90 of longitude per cell."""
    lat_1d = np.array([-30.0, 0.0, 30.0])
    lon_1d = np.array([0.0, 90.0, 180.0, 270.0])
    lat_b_1d = np.array([-45.0, -15.0, 15.0, 45.0])
    lon_b_1d = np.array([-45.0, 45.0, 135.0, 225.0, 315.0])

    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
    lon_b2d, lat_b2d = np.meshgrid(lon_b_1d, lat_b_1d)
    return xr.Dataset(
        coords={
            "lat": (("y", "x"), lat2d),
            "lon": (("y", "x"), lon2d),
            "lat_b": (("y_b", "x_b"), lat_b2d),
            "lon_b": (("y_b", "x_b"), lon_b2d),
        }
    )


@pytest.fixture
def curvilinear():
    """A nonseparable grid, where both bounds vary along both dimensions."""
    y_b = np.linspace(-60.0, 60.0, NY + 1)
    x_b = np.linspace(0.0, 270.0, NX + 1)
    lon_b2d, lat_b2d = np.meshgrid(x_b, y_b)
    # Converge the rows towards each other as the tripolar fold does. The tilt
    # has to be a product rather than a sum: adding a function of `lon_b` alone
    # cancels in the row-to-row difference, which would leave the cell heights
    # separable after all.
    lat_b2d = lat_b2d * (1.0 + 0.005 * lon_b2d)
    lon_b2d = lon_b2d * (1.0 + 0.002 * lat_b2d)

    y = np.linspace(-40.0, 40.0, NY)
    x = np.linspace(45.0, 225.0, NX)
    lon2d, lat2d = np.meshgrid(x, y)
    return xr.Dataset(
        coords={
            "lat": (("y", "x"), lat2d),
            "lon": (("y", "x"), lon2d),
            "lat_b": (("y_b", "x_b"), lat_b2d),
            "lon_b": (("y_b", "x_b"), lon_b2d),
        }
    )


def _static(dxt=None, dyt=None, drop=()):
    """A miniature MOM6 `ocean_static`, on the tracer grid."""
    shape = (NY, NX)
    data = {
        "dxt": (("yh", "xh"), np.full(shape, 1000.0) if dxt is None else dxt),
        "dyt": (("yh", "xh"), np.full(shape, 2000.0) if dyt is None else dyt),
        "areacello": (("yh", "xh"), np.full(shape, 2e6)),
    }
    for name in drop:
        data.pop(name)
    return xr.Dataset(data)


# --- deriving on a rectilinear grid --------------------------------------------


def test_derived_metrics_match_the_arcs_they_stand_for(rectilinear):
    """30 degrees of latitude is the same arc everywhere; 90 of longitude is not."""
    metrics = rectilinear_grid_metrics(rectilinear)

    degree = EARTH_RADIUS_M * np.deg2rad(1.0)
    np.testing.assert_allclose(metrics["dy"].values, 30.0 * degree)

    expected_dx = 90.0 * degree * np.cos(np.deg2rad([-30.0, 0.0, 30.0]))
    np.testing.assert_allclose(
        metrics["dx"].values, np.repeat(expected_dx[:, None], NX, axis=1)
    )


def test_derived_metrics_land_on_the_cell_centers(rectilinear):
    """Bounds are one wider than centers, so the result must come back narrower."""
    metrics = rectilinear_grid_metrics(rectilinear)

    for name in ("dx", "dy"):
        assert metrics[name].dims == ("y", "x")
        assert metrics[name].shape == (NY, NX)


def test_derived_metrics_roughly_reproduce_the_cell_area(rectilinear):
    """dx*dy is not the exact spherical area, but it must not be far off."""
    metrics = rectilinear_grid_metrics(rectilinear)
    product = (metrics["dx"] * metrics["dy"]).values

    lat_b = np.deg2rad(rectilinear["lat_b"].values[:, 0])
    lon_b = np.deg2rad(rectilinear["lon_b"].values[0, :])
    exact = (
        EARTH_RADIUS_M**2
        * (np.sin(lat_b[1:]) - np.sin(lat_b[:-1]))[:, None]
        * np.diff(lon_b)[None, :]
    )

    np.testing.assert_allclose(product, exact, rtol=0.05)


def test_curvilinear_bounds_are_refused(curvilinear):
    """A wrong width is worse than an error, so refuse rather than derive."""
    with pytest.raises(ValueError, match="grid is curvilinear"):
        rectilinear_grid_metrics(curvilinear)


def test_the_curvilinear_fixture_would_actually_give_wrong_widths(curvilinear):
    """Guards the test above: the refusal has to be preventing something.

    If the tilted bounds happened to derive correctly, the check would be
    rejecting a grid it did not need to.
    """
    lat_b = np.deg2rad(curvilinear["lat_b"].values)
    naive_dy = EARTH_RADIUS_M * (lat_b[1:, :-1] - lat_b[:-1, :-1])

    # A separable grid gives the same height for every cell in a row. This one
    # cannot, which is the property that makes the derivation invalid.
    assert naive_dy.std(axis=1).max() > 1.0


# --- carrying the native metrics -----------------------------------------------


def test_native_metrics_come_from_the_model(rectilinear):
    metrics = native_grid_metrics(_static())

    np.testing.assert_array_equal(metrics["dx"].values, np.full((NY, NX), 1000.0))
    np.testing.assert_array_equal(metrics["dy"].values, np.full((NY, NX), 2000.0))
    assert metrics["dx"].attrs["units"] == "m"
    assert metrics["dy"].attrs["units"] == "m"


def test_native_metrics_are_promoted_to_float64():
    """MOM6 ships these as float32; `areacello` is carried at float64."""
    static = _static(dxt=np.full((NY, NX), 1000.0, dtype="float32"))

    assert native_grid_metrics(static)["dx"].dtype == np.float64


def test_a_grid_file_without_metrics_fails_loudly():
    with pytest.raises(ValueError, match="carries no"):
        native_grid_metrics(_static(drop=("dxt",)))


def test_metrics_on_a_different_grid_than_the_area_fail_loudly():
    """Metrics at velocity points would misplace every width by half a cell."""
    static = _static()
    static["dxt"] = (("yq", "xh"), np.ones((NY, NX)))

    with pytest.raises(ValueError, match="same grid as the area"):
        native_grid_metrics(static)
