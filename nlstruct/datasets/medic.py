from collections import defaultdict

from nlstruct.datasets.base import Terminology
import pandas as pd


class MEDIC(Terminology):
    def __init__(self,
                 path,
                 semantic_type="Disease",
                 try_improve_case=True,
                 build_synonym_concepts_mapping=True,
                 synonym_preprocess_fn=None,
                 do_unidecode=False,
                 subs=(),
                 drop_concepts=(),):
        medic = pd.read_csv(
            path,
            comment='#',
            names=['DiseaseName', 'DiseaseID', 'AltDiseaseIDs', 'Definition', 'ParentIDs', 'TreeNumbers', 'ParentTreeNumbers', 'Synonyms', 'SlimMappings'],
            sep='\t',
            # usecols=['DiseaseName', 'DiseaseID', 'Synonyms', 'SlimMappings']
        )
        alt_disease_to_concept = {}
        for row in medic.itertuples():
            if isinstance(row.AltDiseaseIDs, str):
                for alt_disease_id in [disease for disease in row.AltDiseaseIDs.split("|")]:
                    alt_disease_to_concept[alt_disease_id.replace("MESH", "MSH")] = row.DiseaseID.replace("MESH", "MSH")
        concept_synonym_pairs = defaultdict(lambda: {})
        for row in medic.itertuples():
            synonyms = [
                row.DiseaseName,
                *(synonym for synonym in (row.Synonyms.split("|") if isinstance(row.Synonyms, str) else ()))
            ]
            if try_improve_case:
                synonyms = [self.try_improve_case(syn) for syn in synonyms]
            concept = row.DiseaseID.replace("MESH", "MSH")
            concept = alt_disease_to_concept.get(concept, concept)
            concept_synonym_pairs[concept].update(dict.fromkeys(synonyms))

        for concept in drop_concepts:
            del concept_synonym_pairs[concept]

        super().__init__(
            concept_synonym_pairs={concept: list(synset) for concept, synset in concept_synonym_pairs.items()},
            concept_mapping=alt_disease_to_concept,
            concept_semantic_types=defaultdict(lambda: semantic_type),
            build_synonym_concepts_mapping=build_synonym_concepts_mapping,
            synonym_preprocess_fn=synonym_preprocess_fn,
            do_unidecode=do_unidecode,
            subs=subs,
        )

    def try_improve_case(self, text):
        if len(text.split(" ")) > 1 and text == text.upper():
            return text.lower()
        if text == text.capitalize():
            return text.lower()
        return text
