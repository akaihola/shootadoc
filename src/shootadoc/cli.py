#!/usr/bin/env python

from math import log2
from typing import Callable, Optional, Tuple, Union, cast

import click
import PIL.Image
import PIL.ImageMath
from PIL.Image import Image
from PIL.ImageChops import darker, lighter


def _normalize_offset(offset: int, size: int) -> int:
    return offset if offset >= 0 else size - 1


class SlicableImage(Image):
    def _get_absolute_range(
        self, item: Union[slice, int], axis: int
    ) -> Tuple[int, int]:
        size = self.size[axis]
        if item is None:
            return 0, size
        if isinstance(item, slice):
            assert item.step is None
            return (
                0 if item.start is None else _normalize_offset(item.start, size),
                size if item.stop is None else _normalize_offset(item.stop, size),
            )
        offset = _normalize_offset(item, size)
        return offset, offset + 1

    def __getitem__(
        self, item: Tuple[Union[slice, int, None], Union[slice, int, None]]
    ) -> Image:
        x, y = item
        x1, x2 = self._get_absolute_range(x, 0)
        y1, y2 = self._get_absolute_range(y, 1)
        return self.crop((x1, y1, x2, y2))

    @classmethod
    def from_image(cls, image: Image) -> "SlicableImage":
        slicable = cast(SlicableImage, image.copy())
        slicable.__class__ = cls
        return slicable

    @classmethod
    def new(cls, mode: str, size: Tuple[int, int]) -> "SlicableImage":
        image = PIL.Image.new(mode, size)
        image.__class__ = cls
        return image


def get_extreme_neighbor(
    image: SlicableImage, shift: int, aggregate=lighter
) -> Image:
    orig = image[:-shift, :-shift]
    up = image[:-shift, shift:]
    left = image[shift:, :-shift]
    diag = image[shift:, shift:]
    return SlicableImage.from_image(
        aggregate(aggregate(orig, up), aggregate(left, diag))
    )


def fill(image: SlicableImage, direction: int, x: int = None, y: int = None) -> None:
    def get_filler_dimension(offset: Optional[int], size: int) -> int:
        if offset is None:
            return size
        return offset if direction == -1 else size - offset - 1

    def get_filler_offset(offset: Optional[int]) -> int:
        return 0 if offset is None or direction == -1 else offset + 1

    filler = image[x, y].resize(
        (get_filler_dimension(x, image.width), get_filler_dimension(y, image.height))
    )
    image.paste(filler, (get_filler_offset(x), get_filler_offset(y)))


def get_extreme(
    image: SlicableImage, steps: int, mode: Callable[[Image, Image], Image]
) -> Image:
    out = SlicableImage.new(image.mode, image.size)
    assert steps > 0
    for step in range(steps):
        shift = 2 ** step
        image = get_extreme_neighbor(image, shift, mode)
    out.paste(image, (shift, shift))
    fill(out, direction=-1, y=shift)
    fill(out, direction=1, y=out.height - shift)
    fill(out, direction=-1, x=shift)
    fill(out, direction=1, x=out.width - shift)
    return out


@click.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("-b", "--block-size", default=0)
@click.option("-w", "--white-level", default=192)
def handle_image(input_path, output_path, block_size, white_level):
    image = SlicableImage.from_image(PIL.Image.open(input_path).convert("L"))
    if not block_size:
        block_size = int(log2(min(image.size))) - 1
    adjusted_image = PIL.ImageMath.eval(
        "255 * float(image - darkest) / float(brightest - darkest) / gain",
        image=image,
        darkest=get_extreme(image, block_size, PIL.ImageChops.darker),
        brightest=get_extreme(image, block_size, PIL.ImageChops.lighter),
        gain=white_level / 255.0,
    )
    adjusted_image.convert("L").save(output_path)


if __name__ == "__main__":
    handle_image()
