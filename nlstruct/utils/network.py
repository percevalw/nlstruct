import enum
import os
import shutil
import urllib.request

import tqdm as tqdm
from sklearn.datasets._base import _sha256

from nlstruct.environment import root


class NetworkLoadMode(enum.Enum):
    AUTO = 0
    CACHE_ONLY = 1
    FETCH_ONLY = 2


def download_file(url, path, checksum=None):
    bar = None

    def reporthook(_, chunk, total):
        nonlocal bar
        if bar is None:
            bar = tqdm.tqdm(desc=os.path.basename(url), total=total, unit="B", unit_scale=True)
        bar.update(min(chunk, bar.total - bar.n))

    urllib.request.urlretrieve(url, filename=path, reporthook=reporthook)
    if bar is not None:
        bar.close()
    if checksum is not None:
        computed_checksum = _sha256(path)
        if computed_checksum != checksum:
            raise IOError("{} has an SHA256 checksum ({}) "
                          "differing from expected ({}), "
                          "file may be corrupted.".format(path, computed_checksum,
                                                          checksum))


def ensure_files(path, remotes, mode):
    os.makedirs(path, exist_ok=True)
    file_paths = []
    for remote in remotes:
        file_path = path / remote.filename
        file_exist = os.path.exists(str(file_path))
        tmp_file_path = root.tmp(remote.filename)
        if not file_exist and mode == NetworkLoadMode.CACHE_ONLY:
            raise IOError("Could not find cached file {} in {}".format(file_path, tmp_file_path, path))
        elif mode == NetworkLoadMode.FETCH_ONLY or (not file_exist and mode == NetworkLoadMode.AUTO):
            download_file(remote.url, tmp_file_path, remote.checksum)
            shutil.copy(tmp_file_path, file_path)
            os.remove(tmp_file_path)
            # os.rename(tmp_file_path, file_path)
        file_paths.append(file_path)
    return file_paths
