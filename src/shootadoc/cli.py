#!/usr/bin/env python
from dataclasses import dataclass
from typing import Union, Tuple

import click
import numpy as np
from PIL.Image import Image, fromarray as pil_image_fromarray, open as pil_image_open
from PIL.ImageChops import lighter, darker


@dataclass
class ImageSlicer:
    image: Image

    def _normalize_offset(self, offset: int, size: int) -> int:
        return offset if offset >= 0 else size - 1

    def _get_absolute_range(
        self, item: Union[slice, int], axis: int
    ) -> Tuple[int, int]:
        size = self.image.size[axis]
        if isinstance(item, slice):
            assert item.step is None
            return (
                0 if item.start is None else self._normalize_offset(item.start, size),
                size if item.stop is None else self._normalize_offset(item.stop, size),
            )
        offset = self._normalize_offset(item, size)
        return offset, offset + 1

    def __getitem__(self, item: Tuple[Union[slice, int], Union[slice, int]]) -> Image:
        x, y = item
        x1, x2 = self._get_absolute_range(x, 0)
        y1, y2 = self._get_absolute_range(y, 1)
        return self.image.crop((x1, y1, x2, y2))


def get_brightest_neighbor(image: Image, shift: int, aggregate=lighter) -> Image:
    slicer = ImageSlicer(image)
    orig = slicer[:-shift, :-shift]
    down = slicer[:-shift, shift:]
    right = slicer[shift:, :-shift]
    diag = slicer[shift:, shift:]
    return aggregate(aggregate(orig, down), aggregate(right, diag))


def get_extreme(image: Image, steps: int, mode=lighter) -> np.ndarray:
    for step in range(steps):
        shift = 2 ** step
        image = get_brightest_neighbor(image, shift, mode)
    return np.pad(np.array(image), 2 ** (steps - 1), "edge")[:-1, :-1]


@click.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("-b", "--block-size", default=8)
@click.option("-w", "--white-level", default=192)
def handle_image(input_path, output_path, block_size, white_level):
    image = pil_image_open(input_path).convert("L")
    background = get_extreme(image, block_size, lighter)
    darkest = get_extreme(image, block_size, darker)
    binary_image = (np.array(image) - darkest) / (background - darkest + 0.00001) * 255
    binary_image[binary_image < 0] = 0
    binary_image[binary_image > white_level] = 255
    pil_image_fromarray(binary_image).convert("L").save(output_path)


if __name__ == "__main__":
    handle_image()
