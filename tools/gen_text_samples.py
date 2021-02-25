#!/usr/bin/env python

import os.path
import pathlib
import random
import sys
import tempfile
from typing import Sequence, TextIO, Union

import click
import fontTools.ttLib
import PIL.ImageFont
import tqdm
from trdg.generators import GeneratorFromStrings


################################################################################

DISTORSION_ORIENTATION = 2

################################################################################

@click.command()
@click.option("--dict-file", required=True,
              type=click.Path(exists=True, file_okay=True))
@click.option("--fonts-dir", required=True,
              type=click.Path(exists=True, dir_okay=True))
@click.option("--output-dir", default="text_samples", type=click.Path())
@click.option("-b", "--background", "backgrounds",
              multiple=True, type=int, default=(1,),
              help="What kind of background to use. 0: Gaussian Noise,"
                   " 1: Plain white, 2: Quasicrystal, 3: Image. This option"
                   " can be specified multiple times")
@click.option("-d", "--distortion", "distortions",
              multiple=True, type=int, default=(0,),
              help="Distortion applied to the resulting image. 0: None,"
                   " 1: Sine wave, 2: Cosine wave, 3: Random. This option"
                   " can be specified multiple times")
@click.option("-k", "--skew-angle", type=int, default=0,
              help="Skewing angle of the generated text. In positive degrees.")
@click.option("-s", "--size", "sizes",
              required=True, multiple=True, type=int,
              help="Define the height of the produced images if horizontal,"
                   " else the width")
@click.option("--samples", default=500, type=int,
              help="Number of samples for each configuration")
@click.option("--seed", type=int, default=0)
def main(dict_file, fonts_dir, output_dir,
         backgrounds: Sequence[int],
         distortions: Sequence[int],
         skew_angle: int,
         sizes: Sequence[int],
         samples: int,
         seed):
    output_dir = pathlib.Path(output_dir)

    # Filter configurations
    backgrounds = sorted(set(backgrounds))
    distortions = sorted(set(distortions))
    sizes = sorted(set(sizes))

    print("Configuration")
    print("  - backgrounds:", ", ".join(map(str, backgrounds)))
    print("  - distortions:", ", ".join(map(str, distortions)))
    print("  - skewing_angle:", skew_angle)
    print("  - sizes:", ", ".join(map(str, sizes)))

    words, alphabet = _load_dictionary(dict_file)
    fonts = _load_fonts(fonts_dir, alphabet)

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    print("Using seed:", seed)
    random.seed(seed)

    # Run generator
    lexicon_path = output_dir / "lexicon.txt"
    with open(lexicon_path, "wt", encoding="utf-8") as lexicon_fh:
        for background in backgrounds:
            for dist_t in distortions:
                for size in sizes:
                    random.shuffle(words)
                    generator = GeneratorFromStrings(
                        words,
                        count=samples,
                        fonts=fonts,
                        # language="en",
                        size=size,
                        background_type=background,
                        distorsion_type=dist_t,
                        distorsion_orientation=DISTORSION_ORIENTATION,
                        text_color="#6e6e6e,#000000",
                        skewing_angle=skew_angle,
                        random_skew=True,
                        margins=(3,3,3,3))

                    generator_iter = iter(tqdm.tqdm(generator,
                                          total=samples))

                    _run_generator(generator_iter, output_dir, lexicon_fh)
                    lexicon_fh.flush()

    sys.exit(0)


def _run_generator(generator: GeneratorFromStrings,
                   output_dir: Union[str, pathlib.Path],
                   lexicon_fh: TextIO):
    try:
        while True:
            try:
                img, lbl = next(generator)

                # Separate in subdirectories to avoid cluttering
                # the root directory
                try:
                    img_out_dir = output_dir / lbl[0]
                    img_out_dir.mkdir(parents=True, exist_ok=True)
                    subdir = lbl[0]
                except OSError:
                    img_out_dir = output_dir / "#"
                    img_out_dir.mkdir(parents=True, exist_ok=True)
                    subdir = "#"

                with tempfile.NamedTemporaryFile(
                        mode="w+b", prefix="", suffix=".jpg",
                        dir=img_out_dir, delete=False) as tmpf:
                    img.save(tmpf)
                print(f"{subdir}/{os.path.basename(tmpf.name)}\t{lbl}",
                      file=lexicon_fh)

            except OSError:
                pass
    except StopIteration:
        pass


# Load utilities
################################################################################

def _load_dictionary(path):
    print("Loding dictionary ...")
    try:
        with open(path, "rt", encoding="utf-8") as ofh:
            words = [x for x in map(str.strip, ofh) if x]
    except UnicodeDecodeError:
        print("Error reading dictionary file, please make sure the file"
              " encoding is utf-8")
        sys.exit(1)

    alphabet = set()
    for word in words:
        for char in word:
            alphabet.add(char)
    alphabet = sorted(alphabet)

    print("Loaded", len(words), "words")
    print("Alphabet size:", len(alphabet))
    print("Alphabet:", alphabet)

    return words, alphabet


def _load_fonts(fonts_dir: Union[str, pathlib.Path], alphabet: str):
    fonts_dir = pathlib.Path(fonts_dir)

    print("Looking for fonts ...")
    all_fonts = [str(x) for x in fonts_dir.rglob("*.ttf")]
    print("Found", len(all_fonts), "fonts")

    fonts = [x for x in tqdm.tqdm(all_fonts, desc="Filtering fonts")
             if _font_supports_alphabet(x, alphabet)]
    print("# Fonts that support the alphabet:", len(fonts))

    return fonts


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
