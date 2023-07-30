def _get_brightest_neighbor(image, shift, aggregate=ImageChops.lighter):
    w, h = image.size
    orig = image.crop((0, 0, w - shift, h - shift))
    down = image.crop((0, shift, w - shift, h))
    right = image.crop((shift, 0, w, h - shift))
    diag = image.crop((shift, shift, w, h))
    return aggregate(aggregate(orig, down), aggregate(right, diag))


def _get_extreme(image, steps, mode=ImageChops.lighter):
    """


    +--+----+--+
    |  |    |  |
    +--+----+--+
    |  |    |  |
    |  |    |  |
    +--+----+--+
    |  |    |  |
    +--+----+--+

    """
    w, h = image.size
    out = Image.new(image.mode, (w, h))
    for step in range(steps):
        shift = 2 ** step
        image = get_brightest_neighbor(image, shift, mode)
    out.paste(image, (shift, shift))
    top = image.resize((w, shift), Image.NEAREST, (0, 0, w, 1))
    out.paste(top, (shift, 0))
    bottom = image.resize((w, shift - 1), Image.NEAREST, (0, h - 1, w, h))
    out.paste(bottom, (shift, out.height - shift))
    left = out.resize(
        (shift, out.height), Image.NEAREST, (shift, 0, shift + 1, out.height)
    )
    out.paste(left, (0, 0))
    right = out.resize(
        (shift - 1, out.height), Image.NEAREST, (shift - 1, 0, shift, out.height)
    )
    out.paste(right, (out.width - shift + 1, 0))
    return out




@click.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("-b", "--block-size", default=8)
@click.option("-w", "--white-level", default=192)
def threshold(input_path, output_path, block_size, white_level):
    image = Image.open(input_path).convert("L")
    background = get_extreme(image, block_size, "max")
    darkest = get_extreme(image, block_size, "min")
    binary_image = (image - darkest) / (background - darkest + 0.00001) * 255
    binary_image[binary_image < 0] = 0
    binary_image[binary_image > white_level] = 255
    Image.fromarray(binary_image).convert("L").save(output_path)
