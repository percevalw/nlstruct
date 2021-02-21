from collections import defaultdict

from pyner.dataloaders.base import Terminology
import pandas as pd


class MEDIC(Terminology):
    def __init__(self, path, semantic_type="Disease", try_improve_case=True, build_synonym_concepts_mapping=True):
        medic = pd.read_csv(
            path,
            comment='#',
            names=['DiseaseName', 'DiseaseID', 'AltDiseaseIDs', 'Definition', 'ParentIDs', 'TreeNumbers', 'ParentTreeNumbers', 'Synonyms', 'SlimMappings'],
            # usecols=['DiseaseName', 'DiseaseID', 'Synonyms', 'SlimMappings']
        )
        if try_improve_case:
            concept_synonym_pairs = {
                row.DiseaseID.replace("MESH", "MSH"): (
                    self.try_improve_case(row.DiseaseName), *(self.try_improve_case(synonym) for synonym in (row.Synonyms.split("|") if isinstance(row.Synonyms, str) else ())))
                for row in medic.itertuples()
            }
        else:
            concept_synonym_pairs = {
                row.DiseaseID.replace("MESH", "MSH"): (row.DiseaseName, *(synonym for synonym in (row.Synonyms.split("|") if isinstance(row.Synonyms, str) else ())))
                for row in medic.itertuples()
            }
        alt_disease_to_concept = {}
        for row in medic.itertuples():
            if isinstance(row.AltDiseaseIDs, str):
                for alt_disease_id in [disease for disease in row.AltDiseaseIDs.split("|")]:
                    alt_disease_to_concept[alt_disease_id.replace("MESH", "MSH")] = row.DiseaseID.replace("MESH", "MSH")

        super().__init__(
            concept_synonym_pairs=concept_synonym_pairs,
            concept_mapping=alt_disease_to_concept,
            concept_semantic_types=defaultdict(lambda: semantic_type),
            build_synonym_concepts_mapping=build_synonym_concepts_mapping,
        )

    def try_improve_case(self, text):
        if len(text.split(" ")) > 1 and text == text.upper():
            return text.lower()
        if text == text.capitalize():
            return text.lower()
        return text
