import random
import pathlib
from typing import Optional, Tuple, Union, Sequence

import click
import tqdm.auto as tqdm

import cv2
import numpy as np


def random_color():
    return tuple(np.random.randint(256, size=3).tolist())


def draw_random_line_points(img,
                            color: Union[int, Tuple[int, int, int]] = 1,
                            min_thickness: int = 1,
                            max_thickness: int = 30):
    """
    Draws a randomly generated line on top of the given image

    Paramteres
    ----------
    img: np.ndarray
        Base image
    color: Union[int, Tuple[int, int, int]], default 1
        The color of the line. It can either be an int or a tuple of
        three integers representing a BGR color
    min_thickness: int, default 1
        Minimum thickness of the generated line
    max_thickness: int, default 30
        Maximum thickness of the generated line

    Retunrs
    -------
    np.ndarray
        Base image with the drawn line
    """
    img = img.copy()
    h, w = img.shape[:2]

    x1, x2 = np.random.randint(0, w, size=(2,))
    y1, y2 = np.random.randint(0, h, size=(2,))
    t = np.random.randint(min_thickness, max_thickness)

    cv2.line(img, (x1, y1), (x2, y2), color, t)

    return img


def load_artifact(artifact_path: pathlib.Path) -> np.ndarray:
    artifact = cv2.imread(str(artifact_path))
    _, artifact = cv2.threshold(artifact, 128, 255, cv2.THRESH_BINARY)

    h, w = artifact.shape[:2]
    ty = np.random.randint(int(h * .7), h - int(h * .1))
    factor = -1 if random.random() < .5 else 1
    ty = ty * factor

    tfm = np.array([[1, 0, 0], [0, 1, ty]], dtype='float32')
    return cv2.warpAffine(artifact, tfm, (w, h), borderValue=(255, 255, 255))


def merge_random_artifact(im_path: pathlib.Path,
                          artifacts: Sequence[pathlib.Path]):

    im = cv2.imread(str(im_path))

    artifact_path = random.choice(artifacts)
    artifact = load_artifact(artifact_path)
    artifact = cv2.resize(artifact, im.shape[:2][::-1])

    merged = np.minimum(im, artifact)

    return merged


@click.command()
@click.option('--input-path', type=click.Path(exists=True, file_okay=False),
              help='Base directory containing all the input images.',
              required=True)
@click.option('--output-path', type=click.Path(file_okay=False),
              help='Directory where to store the noised images.',
              required=True)
@click.option('--artifacts-path', type=click.Path(file_okay=False),
              help='Path of the artifacts. If not set, the generator will '
                   'only use random lines to add noise.',
              default=None)
@click.option('--min-thickness', type=int, default=1,
              help='Minimum thickness of the noising lines')
@click.option('--max-thickness', type=int, default=5,
              help='Maximum thickness of the noising lines')
def run(input_path: str, output_path: str, artifacts_path: Optional[str],
        min_thickness: int, max_thickness: int):

    input_path = pathlib.Path(input_path)

    output_path = pathlib.Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    if artifacts_path is not None:
        artifacts_path = pathlib.Path(artifacts_path)
        if not artifacts_path.exists():
            raise ValueError('Artifacts path {str(artifacts_path)} '
                             'does not exist')

        artifacts = list(artifacts_path.glob('*/*.png'))
    else:
        artifacts = []

    for im_path in tqdm.tqdm(list(input_path.iterdir())):
        if artifacts:
            im = merge_random_artifact(im_path, artifacts)
        im = draw_random_line_points(im,
                                     color=random_color(),
                                     min_thickness=min_thickness,
                                     max_thickness=max_thickness)
        cv2.imwrite(str(output_path / im_path.name), im)


if __name__ == '__main__':
    run()
