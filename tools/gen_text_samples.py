#!/usr/bin/env python

import os.path
import pathlib
import random
import sys
import tempfile

import click
import fontTools.ttLib
import PIL.ImageFont
import tqdm
from trdg.generators import GeneratorFromStrings


###############################################################################

@click.command()
@click.option("-d", "--dictionary-file", required=True,
              type=click.Path(exists=True, file_okay=True))
@click.option("-f", "--fonts-dir", required=True,
              type=click.Path(exists=True, dir_okay=True))
@click.option("-o", "--output-dir", default="text_samples", type=click.Path())
@click.option("-s", "--seed", type=int, default=0)
def main(dictionary_file, fonts_dir, output_dir, seed):
    fonts_dir = pathlib.Path(fonts_dir)
    output_dir = pathlib.Path(output_dir)

    print("Looking for fonts ...")
    all_fonts = [str(x) for x in fonts_dir.rglob("*.ttf")]
    print("Found", len(all_fonts), "fonts")

    print("Loding dictionary ...")
    try:
        words = _load_dictionary(dictionary_file)
    except UnicodeDecodeError:
        print("Error reading dictionary file, please make sure the file"
              " encoding is utf-8")
    print("Loaded", len(words), "words")

    alphabet = set()
    for word in words:
        for char in word:
            alphabet.add(char)
    alphabet = sorted(alphabet)
    print("Alphabet size:", len(alphabet))
    print("Alphabet:", alphabet)

    fonts = [x for x in tqdm.tqdm(all_fonts, desc="Filtering fonts")
             if _font_supports_alphabet(x, alphabet)]
    print("# Fonts that support the alphabet:", len(fonts))

    # Text generator configurations  (TODO: expose this options to the user)
    backgrounds = [0, 1, 2]
    distorsion_types = [0, 2]
    distorsion_orientation = 2
    sizes = [48, 38, 32]
    num_images_per_conf = 500

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    print("Using seed:", seed)
    random.seed(seed)

    # Run generator
    with open(output_dir / "lexicon.txt", "wt", encoding="utf-8") as lexicon_fh:
        for background in backgrounds:
            for dist_t in distorsion_types:
                for size in sizes:
                    random.shuffle(words)
                    generator = GeneratorFromStrings(
                        words,
                        count=num_images_per_conf,
                        fonts=fonts,
                        language="es",
                        size=size,
                        background_type=background,
                        distorsion_type=dist_t,
                        distorsion_orientation=distorsion_orientation,
                        text_color="#f5f5f5,#000000",
                        skewing_angle=3,
                        random_skew=True,
                        margins=(3,3,3,3))

                    generator_iter = iter(tqdm.tqdm(generator,
                                          total=num_images_per_conf))

                    _run_generator(generator_iter, output_dir, lexicon_fh)
                    lexicon_fh.flush()

    sys.exit()


def _run_generator(generator, output_dir, lexicon_fh):
    try:
        while True:
            try:
                img, lbl = next(generator)
                with tempfile.NamedTemporaryFile(
                        mode="w+b", prefix="", suffix=".jpg",
                        dir=output_dir, delete=False) as tmpf:
                    img.save(tmpf)
                print(f"{os.path.basename(tmpf.name)}\t{lbl}", file=lexicon_fh)
            except OSError:
                pass
    except StopIteration:
        pass


def _load_dictionary(path):
    with open(path, "rt", encoding="utf-8") as ofh:
        return [x for x in map(str.strip, ofh) if x]


def _font_supports_alphabet(filepath, alphabet):
    """Verify that a font contains a specific set of characters.
    Args:
        filepath: Path to fsontfile
        alphabet: A string of characters to check for.
    """
    if alphabet == '':
        return True
    font = fontTools.ttLib.TTFont(filepath)
    if not all(any(ord(c) in table.cmap.keys()
                   for table in font['cmap'].tables)
               for c in alphabet):
        return False
    font = PIL.ImageFont.truetype(filepath)
    try:
        for character in alphabet:
            font.getsize(character)
    except:  # pylint: disable=bare-except
        return False
    return True


################################################################################

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
