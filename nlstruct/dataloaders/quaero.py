import zipfile

from sklearn.datasets._base import RemoteFileMetadata

from nlstruct.collections.dataset import Dataset
from nlstruct.environment.path import root
from nlstruct.utils.network import ensure_files, NetworkLoadMode
from nlstruct.dataloaders.brat import load_from_brat

remote = RemoteFileMetadata(
    url="https://quaerofrenchmed.limsi.fr/QUAERO_FrenchMed_brat.zip",
    checksum="2cf8b5715d938fdc1cd02be75c4eaccb5b8ee14f4148216b8f9b9e80b2445c10",
    filename="QUAERO_FrenchMed_brat.zip")


def load_quaero(resource_path="quaero"):
    path = root.resource(resource_path)
    [file] = ensure_files(path, [remote], mode=NetworkLoadMode.AUTO)
    zip_ref = zipfile.ZipFile(path / "QUAERO_FrenchMed_brat.zip", "r")
    zip_ref.extractall(path)
    zip_ref.close()
    dataset = Dataset.concat([
        load_from_brat(path / "QUAERO_FrenchMed/corpus/train/EMEA", doc_attributes={"source": "EMEA", "split": "train"}),
        load_from_brat(path / "QUAERO_FrenchMed/corpus/train/MEDLINE", doc_attributes={"source": "MEDLINE", "split": "train"}),
        load_from_brat(path / "QUAERO_FrenchMed/corpus/dev/EMEA", doc_attributes={"source": "EMEA", "split": "dev"}),
        load_from_brat(path / "QUAERO_FrenchMed/corpus/dev/MEDLINE", doc_attributes={"source": "MEDLINE", "split": "dev"}),
        load_from_brat(path / "QUAERO_FrenchMed/corpus/test/EMEA", doc_attributes={"source": "EMEA", "split": "test"}),
        load_from_brat(path / "QUAERO_FrenchMed/corpus/test/MEDLINE", doc_attributes={"source": "MEDLINE", "split": "test"}),
    ])

    labels = dataset["comments"].rename({"comment_id": "label_id", "comment": "cui"}, axis=1)
    labels["cui"] = labels["cui"].apply(lambda x: x.split(" "))
    labels = labels.nlstruct.flatten("cui_id", tile_index=True)
    return Dataset(
        **dataset[["docs", "fragments", "mentions"]],
        labels=labels[["doc_id", "mention_id", "cui_id", "cui"]],
    )
