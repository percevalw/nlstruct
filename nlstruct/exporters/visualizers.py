

def render_with_displacy(dataset,
                         doc_name="docs",
                         fragment_name="fragments",
                         mention_name="mentions",
                         doc_id_colname="doc_id",
                         mention_id_colname="mention_id",
                         text_name="text",
                         label_colname="label",
                         raw=False):
    """
    Visualize a labelled mention dataset with spacy's displacy

    Parameters
    ----------
    dataset: nlstruct.collections.dataset.Dataset
        Expected table/columns are:
        - docs: [doc_id, text]
        - mentions: [doc_id, mention_id, label]
        - fragments: [doc_id, mention_id, begin, end]
    doc_name: str
    fragment_name: str
    mention_name: str
    doc_id_colname: str
    mention_id_colname: str
    text_name: str
    label_colname: str
        Those modify the expected frame name / column names in the input dataset

    Returns
    -------
    IPython.core.display.DisplayHandle
    """
    from spacy import displacy
    to_render = []
    for _, doc in dataset[doc_name].iterrows():
        doc_id = doc[doc_id_colname]
        doc_mentions = dataset[mention_name].query(f"{doc_id_colname} == '{doc_id}'").merge(dataset[fragment_name].query(f"{doc_id_colname} == '{doc_id}'"), on=[doc_id_colname, mention_id_colname])
        to_render.append({
            "title": doc[doc_id_colname],
            "text": doc[text_name],
            "ents": [
                {"start": mention["begin"], "end": mention["end"], "label": mention[label_colname]}
                for _, mention in doc_mentions.sort_values(["begin", "end"]).iterrows()
            ]
        })
    if raw:
        return to_render
    return displacy.render(to_render, manual=True, style="ent")
