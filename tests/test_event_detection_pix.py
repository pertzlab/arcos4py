#!/usr/bin/env python

"""Tests for `arcos_py` package."""
import numpy as np
from numpy.testing import assert_array_equal
from skimage.io import imread

from arcos4py.tools._detect_events import track_events_image


def test_1_growing():
    """Test growing event detection on a simple image."""
    test_img = imread('tests/testdata/pix/1_growing.tif')
    true_img = imread('tests/testdata/pix/1_growing_true.tif')
    test_img = np.where(test_img == 255, 0, 1)
    tracked_img = track_events_image(test_img, eps=2, eps_prev=2, min_clustersize=4, dims="TXY")
    assert_array_equal(tracked_img, true_img)


def test_1_side_wave():
    """Test side wave event detection on a simple image."""
    test_img = imread('tests/testdata/pix/1_side_wave.tif')
    true_img = imread('tests/testdata/pix/1_side_wave_true.tif')
    test_img = np.where(test_img == 255, 0, 1)
    tracked_img = track_events_image(test_img, eps=2, eps_prev=2, min_clustersize=4, dims="TXY")
    assert_array_equal(tracked_img, true_img)


def test_1_split():
    """Test split event detection on a simple image."""
    test_img = imread('tests/testdata/pix/1_split.tif')
    true_img = imread('tests/testdata/pix/1_split_true.tif')
    test_img = np.where(test_img == 255, 0, 1)
    tracked_img = track_events_image(test_img, eps=2, eps_prev=2, min_clustersize=4, dims="TXY")
    assert_array_equal(tracked_img, true_img)


def test_2_colliding():
    """Test colliding event detection on a simple image."""
    test_img = imread('tests/testdata/pix/2_colliding.tif')
    true_img = imread('tests/testdata/pix/2_colliding_true.tif')
    test_img = np.where(test_img == 255, 0, 1)
    tracked_img = track_events_image(test_img, eps=2, eps_prev=2, min_clustersize=4, dims="TXY")
    assert_array_equal(tracked_img, true_img)


def test_2_crossing():
    """Test crossing event detection on a simple image."""
    test_img = imread('tests/testdata/pix/2_crossing.tif')
    true_img = imread('tests/testdata/pix/2_crossing_true.tif')
    test_img = np.where(test_img == 255, 0, 1)
    tracked_img = track_events_image(test_img, eps=2, eps_prev=2, min_clustersize=4, dims="TXY")
    assert_array_equal(tracked_img, true_img)


def test_2_crossing_simple_predictor():
    """Test crossing event detection on a simple image."""
    test_img = imread('tests/testdata/pix/2_crossing.tif')
    true_img = imread('tests/testdata/pix/2_crossingsimple_predictor_true.tif')
    test_img = np.where(test_img == 255, 0, 1)
    tracked_img = track_events_image(test_img, eps=2, eps_prev=2, predictor=True, min_clustersize=4, dims="TXY")
    assert_array_equal(tracked_img, true_img)


def test_2_passing():
    """Test passing event detection on a simple image."""
    test_img = imread('tests/testdata/pix/2_passing.tif')
    true_img = imread('tests/testdata/pix/2_passing_true.tif')
    test_img = np.where(test_img == 255, 0, 1)
    tracked_img = track_events_image(test_img, eps=2, min_clustersize=2, n_prev=2, dims="TXY", predictor=False)
    assert_array_equal(tracked_img, true_img)


def test_3_central_growing():
    """Test central growing event detection on a simple image."""
    test_img = imread('tests/testdata/pix/3_central_growing.tif')
    true_img = imread('tests/testdata/pix/3_central_growing_true.tif')
    test_img = np.where(test_img == 255, 0, 1)
    tracked_img = track_events_image(test_img, eps=2, eps_prev=2, min_clustersize=4, dims="TXY")
    assert_array_equal(tracked_img, true_img)


def test_4_colliding():
    """Test colliding event detection on a simple image."""
    test_img = imread('tests/testdata/pix/4_colliding.tif')
    true_img = imread('tests/testdata/pix/4_colliding_true.tif')
    test_img = np.where(test_img == 255, 0, 1)
    tracked_img = track_events_image(test_img, eps=2, eps_prev=2, min_clustersize=4, n_prev=1, dims="TXY")
    assert_array_equal(tracked_img, true_img)


def test_4_colliding_transportaion():
    """Test colliding event detection on a simple image."""
    test_img = imread('tests/testdata/pix/4_colliding.tif')
    true_img = imread('tests/testdata/pix/4_colliding_transportation.tif')
    test_img = np.where(test_img == 255, 0, 1)
    tracked_img = track_events_image(
        test_img,
        eps=2,
        eps_prev=2,
        min_clustersize=4,
        n_prev=2,
        dims="TXY",
        linking_method="transportation",
        reg=10,
        reg_m=100,
    )
    assert_array_equal(tracked_img, true_img)


def test_7_growing_bars():
    """Test growing bars event detection on a simple image."""
    test_img = imread('tests/testdata/pix/7_growing_bars.tif')
    true_img = imread('tests/testdata/pix/7_growing_bars_true.tif')
    test_img = np.where(test_img == 255, 0, 1)
    tracked_img = track_events_image(test_img, eps=3, eps_prev=3, min_clustersize=4, dims="TXY")
    assert_array_equal(tracked_img, true_img)
