# -*- coding: utf-8 -*-

import pathlib
from typing import Optional, Union

import requests
import tqdm.auto as tqdm


################################################################################

def get_models_path() -> pathlib.Path:
    """Get the project models. If the directory does not exist it gets created.

    Returns
    -------
    models_path : Path
        The project models path ($HOME/.gft/models)
    """
    path = pathlib.Path().home() / ".gft" / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file_from_google_drive(file_id: str,
                                    destination: Union[str, pathlib.Path],
                                    progress: bool = False):
    """Downloads a file from google drive given its ID, provided that the file
    is publicly avaliable.

    This implementation was taken from:
    https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

    Parameters
    ----------
    file_id : str
        Google drive file identifier.
    destination : str or Path
        Path to save the file.
    progress : bool
        Whether to show the download progress or not (default: False).
    """
    base_url = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(base_url,
                           params={ 'id' : file_id },
                           stream=True)
    token = _get_google_drive_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(base_url, params = params, stream=True)

    if progress:
        if "content-length" in response.headers:
            content_length = response.headers["content-length"]
        elif "Content-Length" in response.headers:
            content_length = response.headers["content-length"]
        else:
            content_length = 0
    else:
        content_length = None

    save_response_content(response, destination, content_length)


def _get_google_drive_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response: requests.Response,
                          destination: Union[str, pathlib.Path],
                          content_length: Optional[int] = None):
    """Saves the content of a requests response into the specified destination.

    Parameters
    ----------
    response : requests.Response
        A `requests` response.
    destination : str or Path
        Path to save the response content.
    content_length : int or None
        If specified it should be the expected size of the response content,
        if such value is unknown a value of 0 will display a progress bar with
        no ETA. The default value (None) will not display a progress bar.
    """
    chunk_size = 32768

    pbar = None
    if content_length > 0:
        pbar = tqdm.tqdm(total=content_length, desc="Downloading",
                         unit="B", unit_scale=True, unit_divisor=1024)
    elif content_length == 0:
        pbar = tqdm.tqdm(desc="Downloading",
                         unit="B", unit_scale=True, unit_divisor=1024)

    try:
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(len(chunk))
    finally:
        if pbar is not None:
            pbar.close()
