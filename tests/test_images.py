import numpy as np
from PIL import Image

from metastable.images import convert_mask_to_grayscale, convert_to_pil


def test_float_array_in_unit_range_is_scaled():
    arr = np.full((4, 5, 3), 1.0)
    img = convert_to_pil(arr)
    assert isinstance(img, Image.Image)
    assert img.size == (5, 4)
    assert img.getpixel((0, 0)) == (255, 255, 255)


def test_large_value_array_is_cast():
    arr = np.full((2, 2, 3), 200.0)
    assert convert_to_pil(arr).getpixel((1, 1)) == (200, 200, 200)


def test_uint8_array():
    arr = np.zeros((3, 3, 3), dtype=np.uint8)
    arr[..., 0] = 128
    assert convert_to_pil(arr).getpixel((0, 0)) == (128, 0, 0)


def test_pil_image_passthrough():
    img = Image.new("RGB", (2, 2))
    assert convert_to_pil(img) is img


def test_mask_converted_to_grayscale():
    mask = Image.new("RGB", (2, 2), (255, 255, 255))
    gray = convert_mask_to_grayscale(mask)
    assert gray.mode == "L"
    assert gray.getpixel((0, 0)) == 255


def test_grayscale_mask_passthrough():
    mask = Image.new("L", (2, 2))
    assert convert_mask_to_grayscale(mask) is mask
