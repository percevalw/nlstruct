from pyner.dataloaders.base import Terminology

import pandas as pd
import os
from collections import defaultdict


class UMLS(Terminology):
    def __init__(self, path, try_improve_case=True, build_synonym_concepts_mapping=True, debug=False, preferred_sabs=(
          "SNOMEDCT_US",
          "MSH",
          "MEDCIN",
          "NCI",
          "MTH"
    ), preferred_lat=('ENG', 'FRE'), preferred_tty=('PTN', 'PN', 'PT', 'RPT')):
        mrconso = pd.read_csv(
            os.path.join(path, "MRCONSO.RRF"),
            sep="|",
            header=None,
            index_col=False,
            names=["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF"],
            usecols=["CUI", "STR", "LAT", "SAB", "CODE", "ISPREF", "TTY", "LAT"],
            nrows=1000 if debug else None,
        )
        mrsty = pd.read_csv(
            os.path.join(path, "MRSTY.RRF"),
            sep="|",
            header=None,
            index_col=False,
            names=["CUI", "TUI", "STN", "STY", "ATUI", "CVF"],
            usecols=["CUI", "STY"],
            nrows=1000 if debug else None,
        )
        semantic_types = dict(zip(mrsty.CUI, mrsty.STY))

        mrconso = mrconso[~(mrconso.STR.isna() | mrconso.CODE.isna())]
        preferred_tty = ['PN', 'PT']
        mrconso['sab_index'] = mrconso['SAB'].apply(lambda x: -preferred_sabs.index(x) if x in preferred_sabs else 1)
        mrconso['tty_index'] = mrconso['TTY'].apply(lambda x: -preferred_tty.index(x) if x in preferred_tty else 1)
        mrconso['is_pref_index'] = (mrconso['IS_PREF'] == 'Y').astype(int)
        mrconso['lang_index'] = mrconso['LAT'].apply(lambda x: -preferred_lat.index(x) if x in preferred_lat else 1)
        mrconso = mrconso.sort_values(["sab_index", "is_pref_index", "lang_index", "tty_index"])
        mrconso = mrconso.drop_duplicates(["CUI", "STR"])

        if try_improve_case:
            mrconso['STR'] = mrconso['STR'].apply(self.try_improve_case)

        concept_synonym_pairs = defaultdict(lambda: [])
        for cui, synonym in zip(mrconso['CUI'].tolist(), mrconso['STR'].tolist()):
            concept_synonym_pairs[cui].append(synonym)
        alt_concept_mrconso = mrconso[~(mrconso.CODE.isna() | mrconso.SAB.isna())]
        concept_mapping = dict(zip(alt_concept_mrconso.SAB.astype(str).str.cat(alt_concept_mrconso.CODE.astype(str), sep=":").tolist(), alt_concept_mrconso.CUI.tolist()))

        super().__init__(
            concept_synonym_pairs=concept_synonym_pairs,
            concept_mapping=concept_mapping,
            concept_semantic_types=semantic_types,
            build_synonym_concepts_mapping=build_synonym_concepts_mapping,
        )

    def try_improve_case(self, text):
        if len(text.split(" ")) > 1 and text == text.upper():
            return text.lower()
        if text == text.capitalize():
            return text.lower()
        return text
