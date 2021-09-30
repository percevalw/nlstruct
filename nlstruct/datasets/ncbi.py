import zipfile

from sklearn.datasets._base import RemoteFileMetadata

from nlstruct.datasets.base import NetworkLoadMode, ensure_files, NormalizationDataset


class NCBI(NormalizationDataset):
    REMOTE_FILES = [
        RemoteFileMetadata(
            url="https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItrainset_corpus.zip",
            checksum="26157233d70aeda0b2ac1dda4fc9369b0717bd888f5afe511d0c1c6a5ad307a0",
            filename="NCBItrainset_corpus.zip"),
        RemoteFileMetadata(
            url="https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItestset_corpus.zip",
            checksum="b978442f39c739deb6619c70e7b07327d9eb1b71aff64996c02a592435583f46",
            filename="NCBItestset_corpus.zip"),
        RemoteFileMetadata(
            url="https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBIdevelopset_corpus.zip",
            checksum="61681ad09356619c8f4f3e1738663dd007ad0135720fdc90195120fea944cbe9",
            filename="NCBIdevelopset_corpus.zip"),
    ]

    def __init__(self, path, terminology=None, map_concepts=False, unmappable_concepts="raise", relabel_with_semantic_type=False, debug=False, preprocess_fn=None):
        train_data, val_data, test_data = self.download_and_extract(path, debug)
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

    def download_and_extract(self, path, debug):
        train_file, test_file, dev_file = ensure_files(path, self.REMOTE_FILES, mode=NetworkLoadMode.AUTO)

        splits = {}
        # Load full datasets with concept annotations
        for split, file in [("train", train_file),
                            ("test", test_file),
                            ("val", dev_file)]:
            docs = []
            zip_ref = zipfile.ZipFile(file, "r")
            zip_ref.extractall(path)
            zip_ref.close()
            with open(str(file).replace(".zip", ".txt"), "r") as cursor:
                entities = []
                doc = {
                    "doc_id": None,
                    "entities": entities
                }  # accumulate sample info here
                counter = -1  # count the line number in the current sample

                for line in cursor.readlines():
                    counter += 1
                    # end of sample, yield it
                    if not line.strip():
                        if doc["doc_id"]:
                            docs.append(doc)
                            entities = []
                            doc = {
                                "doc_id": None,
                                "entities": entities
                            }
                        counter = -1
                    elif counter == 0:
                        sample_id, title = line.split("|t|")
                        doc["doc_id"] = sample_id
                        doc["text"] = title
                    elif counter == 1:
                        doc["text"] = doc["text"] + line.split("|a|")[1].strip("\n")
                    else:
                        _, begin, end, synonym, category, cui_set = line[:-1].split("\t")
                        fragments = [{"begin": int(begin), "end": int(end)}]
                        labels = [c2.strip() for c1 in cui_set.split("|") for c2 in c1.split('+')]
                        sources = ['OMIM' if l.startswith('OMIM:') else "MSH" for l in labels]
                        codes = [l.split(":")[-1] for l in labels]
                        entity = {
                            "entity_id": doc["doc_id"] + "-" + str(len(entities)),
                            "fragments": fragments,
                            "synonym": synonym,
                            "label": category,
                            "concept": tuple(":".join((source, code)) for source, code in zip(sources, codes)),
                        }
                        entities.append(entity)
                if doc["doc_id"] is not None:
                    docs.append(doc)
            splits[split] = docs
        subset = slice(None) if not debug else slice(0, 50)

        train_data = splits["train"][subset]
        val_data = splits["val"][subset]
        test_data = splits["test"]  # Never subset the test set, we don't want to give false hopes

        return train_data, val_data, test_data
