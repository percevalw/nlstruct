from tqdm import tqdm

from nlstruct.environment.cache import get_cache


def to_conll(
      dataset,
      token_cols=("token_orth", "tag"),
      destination=None,
      with_tqdm=False,
      token_name="tokens",
      doc_name="docs",
      doc_id_colname="doc_id",
      token_idx_colname="token_idx",
):
    """

    Parameters
    ----------
    dataset: nlstruct.collections.dataset.Dataset
    token_cols: list of str
        Columns in the `token_name` frame to put in the CoNLL file
    destination: str or pathlib.Path
        Destination (folder) of the .conll and .txt files
        If None: prints the output
    with_tqdm: bool
        Print progress with tqdm
    token_name: str
        Name of the tokens frame
    doc_name:
        Name of the document frame
    doc_id_colname:
        Column name of the document ids
    token_idx_colname
        Column name of the token index in documents
    """
    cache = get_cache(destination) if destination is not None else None
    for doc_id, doc_tokens in tqdm(dataset[token_name].groupby([doc_id_colname], sort="begin")):
        file_opener = open(cache / (doc_id + ".conll"), "w") if cache is not None else memoryview(b'')
        with file_opener as file:
            for (token_idx, *token_vals) in doc_tokens[[token_idx_colname, *token_cols]].itertuples(index=False):  # iter(zip(*df)) is way faster than df.iterrows()
                print(token_idx, "\t", "\t".join(token_vals), file=file if file_opener is not None else None)
    for doc_id, doc_text in dataset[doc_name][[doc_id_colname, "text"]].itertuples(index=False):
        file_opener = open(cache / (doc_id + ".txt"), "w") if cache is not None else memoryview(b'')
        with file_opener as file:
            print(doc_text, file=file if file_opener is not None else None)
