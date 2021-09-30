import os
import random
import zipfile

from sklearn.datasets._base import RemoteFileMetadata

from nlstruct.datasets.brat import load_from_brat
from nlstruct.datasets.base import NetworkLoadMode, ensure_files, NormalizationDataset


class QUAERO(NormalizationDataset):
    REMOTE_FILES = [
        RemoteFileMetadata(
            url="https://quaerofrenchmed.limsi.fr/QUAERO_FrenchMed_brat.zip",
            checksum="2cf8b5715d938fdc1cd02be75c4eaccb5b8ee14f4148216b8f9b9e80b2445c10",
            filename="QUAERO_FrenchMed_brat.zip")
    ]

    def __init__(self, path, terminology=None, sources=("EMEA", "MEDLINE"), version="2016", val_split=None, seed=False, debug=False,
                 map_concepts=False, unmappable_concepts="raise", relabel_with_semantic_type=False, preprocess_fn=None):
        assert version in ("2015", "2016")
        if val_split is not None or seed is not False:
            assert version == "2015", "As validation split already exist for Quaero 2016, leave val_split=None and seed=False"
            val_split = val_split
        if not isinstance(sources, (tuple, list)):
            sources = (sources,)
        self.sources = sources = tuple(sources)
        train_data, val_data, test_data = self.download_and_extract(path, version, sources, val_split, seed, debug)

        super().__init__(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            terminology=terminology,
            map_concepts=map_concepts,
            unmappable_concepts=unmappable_concepts,
            relabel_with_semantic_type=relabel_with_semantic_type,
            preprocess_fn=preprocess_fn,
        )

    def download_and_extract(self, path, version, sources=("EMEA", "MEDLINE"), val_split=False, seed=False, debug=False):
        """
        Loads the Quaero dataset

        Parameters
        ----------
        path: str
            Location of the Quaero files
        version: str
            Version to load, either '2015' or '2016'
        val_split: float
            Will only be used if version is '2015' since no dev set was defined for this version
        seed: int
            Will only be used if version is '2015' since no dev set was defined for this version
        sources: tuple of str
            Which sources to load, ie EMEA, MEDLINE
        Returns
        -------
        Dataset
        """

        [file] = ensure_files(path, self.REMOTE_FILES, mode=NetworkLoadMode.AUTO)
        zip_ref = zipfile.ZipFile(file, "r")
        zip_ref.extractall(path)
        zip_ref.close()
        train_data = [
            *[{**doc, "source": "EMEA", "entities": [{**entity, "concept": tuple(sorted(part.strip() for comment in entity["comments"]
                                                                                 for part in comment["comment"].strip().strip("+").split(" ")))} for entity in doc["entities"]]}
              for doc in load_from_brat(os.path.join(path, "QUAERO_FrenchMed/corpus/train/EMEA"))],
            *[{**doc, "source": "MEDLINE", "entities": [{**entity, "concept": tuple(sorted(part.strip() for comment in entity["comments"]
                                                                                    for part in comment["comment"].strip().strip("+").split(" ")))} for entity in doc["entities"]]}
              for doc in load_from_brat(os.path.join(path, "QUAERO_FrenchMed/corpus/train/MEDLINE"))],
        ]
        train_data = [doc for doc in train_data if doc["source"] in sources]
        val_data = [
            *[{**doc, "source": "EMEA", "entities": [{**entity, "concept": tuple(sorted(part.strip() for comment in entity["comments"]
                                                                                 for part in comment["comment"].strip().strip("+").split(" ")))} for entity in doc["entities"]]}
              for doc in load_from_brat(os.path.join(path, "QUAERO_FrenchMed/corpus/dev/EMEA"))],
            *[{**doc, "source": "MEDLINE", "entities": [{**entity, "concept": tuple(sorted(part.strip() for comment in entity["comments"]
                                                                                    for part in comment["comment"].strip().strip("+").split(" ")))} for entity in doc["entities"]]}
              for doc in load_from_brat(os.path.join(path, "QUAERO_FrenchMed/corpus/dev/MEDLINE"))],
        ]
        val_data = [doc for doc in val_data if doc["source"] in sources]
        test_data = [
            *[{**doc, "source": "EMEA", "entities": [{**entity, "concept": tuple(sorted(part.strip() for comment in entity["comments"]
                                                                                 for part in comment["comment"].strip().strip("+").split(" ")))} for entity in doc["entities"]]}
              for doc in load_from_brat(os.path.join(path, "QUAERO_FrenchMed/corpus/test/EMEA"))],
            *[{**doc, "source": "MEDLINE", "entities": [{**entity, "concept": tuple(sorted(part.strip() for comment in entity["comments"]
                                                                                    for part in comment["comment"].strip().strip("+").split(" ")))} for entity in doc["entities"]]}
              for doc in load_from_brat(os.path.join(path, "QUAERO_FrenchMed/corpus/test/MEDLINE"))],
        ]
        test_data = [doc for doc in test_data if doc["source"] in sources]

        if version == "2015":
            if val_split:
                shuffled_data = list(train_data)
                if seed is not False:
                    random.Random(seed).shuffle(shuffled_data)
                offset = val_split if isinstance(val_split, int) else int(val_split * len(shuffled_data))
                val_data = shuffled_data[:offset]
                train_data = shuffled_data[offset:]
            else:
                val_data = []

        subset = slice(None) if not debug else slice(0, 50)
        train_data = train_data[subset]
        val_data = val_data[subset]
        test_data = test_data  # Never subset the test set, we don't want to give false hopes

        return train_data, val_data, test_data
