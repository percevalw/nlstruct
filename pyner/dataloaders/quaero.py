import os
import random
import zipfile

from sklearn.datasets._base import RemoteFileMetadata

from pyner.dataloaders.brat import load_from_brat
from pyner.dataloaders.base import NetworkLoadMode, ensure_files, NERDataset


class QUAERO(NERDataset):
    REMOTE_FILES = [
        RemoteFileMetadata(
            url="https://quaerofrenchmed.limsi.fr/QUAERO_FrenchMed_brat.zip",
            checksum="2cf8b5715d938fdc1cd02be75c4eaccb5b8ee14f4148216b8f9b9e80b2445c10",
            filename="QUAERO_FrenchMed_brat.zip")
    ]

    def __init__(self, path, sources=("EMEA", "MEDLINE"), version="2016", val_split=None, seed=False, debug=False):
        super().__init__()
        self.path = path
        self.batch_size = None
        self.debug = debug
        self.train_data = None
        self.val_data = None
        self.test_data = None
        assert version in ("2015", "2016")
        if val_split is not None or seed is not False:
            assert version == "2015", "As validation split already exist for Quaero 2016, leave val_split=None and seed=False"
            self.val_split = val_split
        self.seed = seed
        self.version = version

        if not isinstance(sources, (tuple, list)):
            self.sources = (sources,)
        self.sources = tuple(sources)

        self.download_and_extract()

    def download_and_extract(self):
        """
        Loads the Quaero dataset

        Parameters
        ----------
        resource_path: str
            Location of the Quaero files
        version: str
            Version to load, either '2015' or '2016'
        dev_split: float
            Will only be used if version is '2015' since no dev set was defined for this version
        seed: int
            Will only be used if version is '2015' since no dev set was defined for this version
        Returns
        -------
        Dataset
        """

        [file] = ensure_files(self.path, self.REMOTE_FILES, mode=NetworkLoadMode.AUTO)
        zip_ref = zipfile.ZipFile(file, "r")
        zip_ref.extractall(self.path)
        zip_ref.close()
        self.train_data = [
            *[{**doc, "source": "EMEA", "entities": [{**entity, "concept": tuple(part for comment in entity["comments"] for part in comment["comment"].split(" "))} for entity in doc["entities"]]}
              for doc in load_from_brat(os.path.join(self.path, "QUAERO_FrenchMed/corpus/train/EMEA"))],
            *[{**doc, "source": "MEDLINE", "entities": [{**entity, "concept": tuple(part for comment in entity["comments"] for part in comment["comment"].split(" "))} for entity in doc["entities"]]}
              for doc in load_from_brat(os.path.join(self.path, "QUAERO_FrenchMed/corpus/train/MEDLINE"))],
        ]
        self.train_data = [doc for doc in self.train_data if doc["source"] in self.sources]
        self.val_data = [
            *[{**doc, "source": "EMEA", "entities": [{**entity, "concept": tuple(part for comment in entity["comments"] for part in comment["comment"].split(" "))} for entity in doc["entities"]]}
              for doc in load_from_brat(os.path.join(self.path, "QUAERO_FrenchMed/corpus/dev/EMEA"))],
            *[{**doc, "source": "MEDLINE", "entities": [{**entity, "concept": tuple(part for comment in entity["comments"] for part in comment["comment"].split(" "))} for entity in doc["entities"]]}
              for doc in load_from_brat(os.path.join(self.path, "QUAERO_FrenchMed/corpus/dev/MEDLINE"))],
        ]
        self.val_data = [doc for doc in self.val_data if doc["source"] in self.sources]
        self.test_data = [
            *[{**doc, "source": "EMEA", "entities": [{**entity, "concept": tuple(part for comment in entity["comments"] for part in comment["comment"].split(" "))} for entity in doc["entities"]]}
              for doc in load_from_brat(os.path.join(self.path, "QUAERO_FrenchMed/corpus/test/EMEA"))],
            *[{**doc, "source": "MEDLINE", "entities": [{**entity, "concept": tuple(part for comment in entity["comments"] for part in comment["comment"].split(" "))} for entity in doc["entities"]]}
              for doc in load_from_brat(os.path.join(self.path, "QUAERO_FrenchMed/corpus/test/MEDLINE"))],
        ]
        self.test_data = [doc for doc in self.test_data if doc["source"] in self.sources]

        if self.version == "2015":
            shuffled_data = list(self.train_data)
            if self.seed is not False:
                random.Random(self.seed).shuffle(shuffled_data)
            offset = self.val_split if isinstance(self.val_split, int) else int(self.val_split * len(shuffled_data))
            self.val_data = shuffled_data[:offset]
            self.train_data = shuffled_data[offset:]

        subset = slice(None) if not self.debug else slice(0, 50)
        self.train_data = self.train_data[subset]
        self.val_data = self.val_data[subset]
        self.test_data = self.test_data  # Never subset the test set, we don't want to give false hopes
