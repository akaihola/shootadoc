#!/usr/bin/env python

import click
import numpy as np
from PIL import Image, ImageChops, ImageOps


def get_brightest_neighbor(image, shift, mode="max"):
    orig = image[:-shift, :-shift]
    down = image[:-shift, shift:]
    right = image[shift:, :-shift]
    diag = image[shift:, shift:]
    layers = np.array([orig, down, right, diag])
    aggregate = getattr(layers, mode)
    return aggregate(axis=0)


def get_extreme(image, steps, mode="max"):
    for step in range(steps):
        shift = 2 ** step
        image = get_brightest_neighbor(image, shift, mode)
    return np.pad(image, 2 ** (steps - 1), "edge")[:-1, :-1]


@click.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("-b", "--block-size", default=8)
@click.option("-w", "--white-level", default=192)
def handle_image(input_path, output_path, block_size, white_level):
    image = np.array(Image.open(input_path).convert("L"))
    background = get_extreme(image, block_size, "max")
    darkest = get_extreme(image, block_size, "min")
    binary_image = (image - darkest) / (background - darkest + 0.00001) * 255
    binary_image[binary_image < 0] = 0
    binary_image[binary_image > white_level] = 255
    Image.fromarray(binary_image).convert("L").save(output_path)


if __name__ == "__main__":
    handle_image()
