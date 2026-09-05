import random

import cv2
import numpy as np
import pytest
import torch

from src.data.dataset import get_dataloaders
from src.data.generate_synthetic import generate_mock_dataset
from src.data.preprocess import (
    preprocess_for_prediction,
    run_full_preprocessing,
)


@pytest.fixture(autouse=True)
def restore_global_random_state():
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def independent_linear_oracle(image_raw):
    image_hwc = image_raw.astype(np.float32)
    squared = image_hwc ** 2
    filtered = np.stack(
        [cv2.blur(squared[:, :, channel], (5, 5)) for channel in range(2)],
        axis=2,
    )
    db = 10.0 * np.log10(np.clip(filtered, 1e-5, None))
    return (np.clip(db, -25.0, 0.0) + 25.0) / 25.0


def independent_db_oracle(image_db):
    image_hwc = image_db.astype(np.float32)
    linear = 10.0 ** (image_hwc / 10.0)
    filtered = np.stack(
        [cv2.blur(linear[:, :, channel], (5, 5)) for channel in range(2)],
        axis=2,
    )
    db = 10.0 * np.log10(np.clip(filtered, 1e-5, None))
    return (np.clip(db, -25.0, 0.0) + 25.0) / 25.0


def test_positive_linear_input_is_normalized_with_signal_and_patch_alignment():
    image = np.linspace(0.05, 0.8, 32 * 32 * 2, dtype=np.float32).reshape(32, 32, 2)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[:16, :16] = 255

    expected = independent_linear_oracle(image)
    actual = preprocess_for_prediction(image)
    _, norm_without_mask = run_full_preprocessing(image, patch_size=16, stride=16)
    image_patches, mask_patches = run_full_preprocessing(
        image, mask, patch_size=16, stride=16
    )

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(norm_without_mask, expected, rtol=0, atol=1e-6)
    assert np.isfinite(actual).all()
    assert np.ptp(actual) > 0
    assert len(image_patches) == len(mask_patches) == 4
    np.testing.assert_allclose(image_patches[0], expected[:16, :16], rtol=0, atol=1e-6)
    assert np.count_nonzero(mask_patches[0]) > 0
    assert all(np.count_nonzero(patch) == 0 for patch in mask_patches[1:])


def test_negative_db_input_preserves_existing_formula_for_hwc_and_chw():
    image_hwc = np.array(
        [[[-40.0 + row + col, -35.0 + row - col] for col in range(4)] for row in range(4)],
        dtype=np.float32,
    )
    image_hwc[1, 1] = [-5.0, 1.0]
    image_hwc[2, 2] = [5.0, -60.0]
    image_chw = image_hwc.transpose(2, 0, 1)

    expected = independent_db_oracle(image_hwc)
    np.testing.assert_allclose(
        preprocess_for_prediction(image_hwc), expected, rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        preprocess_for_prediction(image_chw), expected, rtol=0, atol=1e-6
    )

    _, norm = run_full_preprocessing(image_chw, patch_size=4, stride=4)
    np.testing.assert_allclose(norm, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    ("category", "random_values", "expected_count"),
    [
        ("oil", [0.009, 0.011, 0.001], 3),
        ("lookalike", [0.009, 0.006, 0.004], 2),
        ("no_oil", [0.002, 0.001, 0.003], 2),
    ],
)
def test_balanced_sampling_retains_positive_patches_and_category_rates(
    monkeypatch, category, random_values, expected_count
):
    image = np.ones((32, 32, 2), dtype=np.float32)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[:16, :16] = 1
    values = iter(random_values)
    monkeypatch.setattr(random, "random", lambda: next(values))

    images, masks = run_full_preprocessing(
        image, mask, patch_size=16, stride=16, category=category
    )

    assert len(images) == len(masks) == expected_count
    assert any(np.count_nonzero(patch) for patch in masks)


def test_generated_dataset_loaders_are_nonempty_finite_bounded_and_channelwise(tmp_path):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    generate_mock_dataset(str(tmp_path), num_scenes=5, size=64)
    loaders = get_dataloaders(str(tmp_path), patch_size=32, stride=32, batch_size=2)

    assert all(len(loader.dataset) > 0 for loader in loaders)
    for loader in loaders:
        images, masks = next(iter(loader))
        assert images.dtype == torch.float32
        assert masks.dtype == torch.float32
        assert images.ndim == 4 and images.shape[1:] == (2, 32, 32)
        assert masks.ndim == 4 and masks.shape[1:] == (1, 32, 32)
        assert torch.isfinite(images).all()
        assert torch.isfinite(masks).all()
        assert float(images.min()) >= 0.0
        assert float(images.max()) <= 1.0
        assert set(torch.unique(masks).tolist()).issubset({0.0, 1.0})
        assert np.ptp(images[:, 0].numpy()) > 0
