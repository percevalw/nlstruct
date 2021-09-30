from nlstruct.datasets.base import Terminology

import pandas as pd
import os
from collections import defaultdict

sty_groups = {'Activity': 'ACTI', 'Behavior': 'ACTI', 'Daily or Recreational Activity': 'ACTI', 'Event': 'ACTI', 'Governmental or Regulatory Activity': 'ACTI', 'Individual Behavior': 'ACTI',
              'Machine Activity': 'ACTI', 'Occupational Activity': 'ACTI', 'Social Behavior': 'ACTI', 'Anatomical Structure': 'ANAT', 'Body Location or Region': 'ANAT',
              'Body Part, Organ, or Organ Component': 'ANAT', 'Body Space or Junction': 'ANAT', 'Body Substance': 'ANAT', 'Body System': 'ANAT', 'Cell': 'ANAT', 'Cell Component': 'ANAT',
              'Embryonic Structure': 'ANAT', 'Fully Formed Anatomical Structure': 'ANAT', 'Tissue': 'ANAT', 'Amino Acid, Peptide, or Protein': 'CHEM', 'Antibiotic': 'CHEM',
              'Biologically Active Substance': 'CHEM', 'Biomedical or Dental Material': 'CHEM', 'Carbohydrate': 'CHEM', 'Chemical': 'CHEM', 'Chemical Viewed Functionally': 'CHEM',
              'Chemical Viewed Structurally': 'CHEM', 'Clinical Drug': 'CHEM', 'Eicosanoid': 'CHEM', 'Element, Ion, or Isotope': 'CHEM', 'Enzyme': 'CHEM', 'Hazardous or Poisonous Substance': 'CHEM',
              'Hormone': 'CHEM', 'Immunologic Factor': 'CHEM', 'Indicator, Reagent, or Diagnostic Aid': 'CHEM', 'Inorganic Chemical': 'CHEM', 'Lipid': 'CHEM',
              'Neuroreactive Substance or Biogenic Amine': 'CHEM', 'Nucleic Acid, Nucleoside, or Nucleotide': 'CHEM', 'Organic Chemical': 'CHEM', 'Organophosphorus Compound': 'CHEM',
              'Pharmacologic Substance': 'CHEM', 'Receptor': 'CHEM', 'Steroid': 'CHEM', 'Vitamin': 'CHEM', 'Classification': 'CONC', 'Conceptual Entity': 'CONC', 'Functional Concept': 'CONC',
              'Group Attribute': 'CONC', 'Idea or Concept': 'CONC', 'Intellectual Product': 'CONC', 'Language': 'CONC', 'Qualitative Concept': 'CONC', 'Quantitative Concept': 'CONC',
              'Regulation or Law': 'CONC', 'Spatial Concept': 'CONC', 'Temporal Concept': 'CONC', 'Drug Delivery Device': 'DEVI', 'Medical Device': 'DEVI', 'Research Device': 'DEVI',
              'Acquired Abnormality': 'DISO', 'Anatomical Abnormality': 'DISO', 'Cell or Molecular Dysfunction': 'DISO', 'Congenital Abnormality': 'DISO', 'Disease or Syndrome': 'DISO',
              'Experimental Model of Disease': 'DISO', 'Finding': 'DISO', 'Injury or Poisoning': 'DISO', 'Mental or Behavioral Dysfunction': 'DISO', 'Neoplastic Process': 'DISO',
              'Pathologic Function': 'DISO', 'Sign or Symptom': 'DISO', 'Amino Acid Sequence': 'GENE', 'Carbohydrate Sequence': 'GENE', 'Gene or Genome': 'GENE', 'Molecular Sequence': 'GENE',
              'Nucleotide Sequence': 'GENE', 'Geographic Area': 'GEOG', 'Age Group': 'LIVB', 'Amphibian': 'LIVB', 'Animal': 'LIVB', 'Archaeon': 'LIVB', 'Bacterium': 'LIVB', 'Bird': 'LIVB',
              'Eukaryote': 'LIVB', 'Family Group': 'LIVB', 'Fish': 'LIVB', 'Fungus': 'LIVB', 'Group': 'LIVB', 'Human': 'LIVB', 'Mammal': 'LIVB', 'Organism': 'LIVB',
              'Patient or Disabled Group': 'LIVB', 'Plant': 'LIVB', 'Population Group': 'LIVB', 'Professional or Occupational Group': 'LIVB', 'Reptile': 'LIVB', 'Vertebrate': 'LIVB', 'Virus': 'LIVB',
              'Entity': 'OBJC', 'Food': 'OBJC', 'Manufactured Object': 'OBJC', 'Physical Object': 'OBJC', 'Substance': 'OBJC', 'Biomedical Occupation or Discipline': 'OCCU',
              'Occupation or Discipline': 'OCCU', 'Health Care Related Organization': 'ORGA', 'Organization': 'ORGA', 'Professional Society': 'ORGA', 'Self-help or Relief Organization': 'ORGA',
              'Biologic Function': 'PHEN', 'Environmental Effect of Humans': 'PHEN', 'Human-caused Phenomenon or Process': 'PHEN', 'Laboratory or Test Result': 'PHEN',
              'Natural Phenomenon or Process': 'PHEN', 'Phenomenon or Process': 'PHEN', 'Cell Function': 'PHYS', 'Clinical Attribute': 'PHYS', 'Genetic Function': 'PHYS', 'Mental Process': 'PHYS',
              'Molecular Function': 'PHYS', 'Organism Attribute': 'PHYS', 'Organism Function': 'PHYS', 'Organ or Tissue Function': 'PHYS', 'Physiologic Function': 'PHYS',
              'Diagnostic Procedure': 'PROC', 'Educational Activity': 'PROC', 'Health Care Activity': 'PROC', 'Laboratory Procedure': 'PROC', 'Molecular Biology Research Technique': 'PROC',
              'Research Activity': 'PROC', 'Therapeutic or Preventive Procedure': 'PROC'}


class UMLS(Terminology):
    def __init__(self, path, try_improve_case=True, build_synonym_concepts_mapping=True, debug=False,
                 preferred_sab=(
                       "MTH",
                       "SNOMEDCT_US",
                       "MSH",
                       "MEDCIN",
                       "NCI",
                       "MTH",
                 ),
                 use_sty_groups=False,
                 preferred_lat=('ENG', 'FRE'),
                 preferred_tty=('FN', 'PTN', 'PN', 'PT', 'RPT', 'SY'),
                 query=None,
                 synonym_preprocess_fn=None,
                 do_unidecode=False,
                 subs=(),):
        global mrconso
        mrconso = pd.read_csv(
            os.path.join(path, "MRCONSO.RRF"),
            sep="|",
            header=None,
            index_col=False,
            names=["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF"],
            usecols=["CUI", "STR", "LAT", "SAB", "CODE", "ISPREF", "TTY"] if query is None else None,
            nrows=1000 if debug else None,
        )
        if query is not None:
            mrconso = mrconso.query(query)
            assert len(mrconso) > 0, "Empty UMLS after query !"
        mrconso = mrconso[["CUI", "STR", "LAT", "SAB", "CODE", "ISPREF", "TTY"]]
        mrsty = pd.read_csv(
            os.path.join(path, "MRSTY.RRF"),
            sep="|",
            header=None,
            index_col=False,
            names=["CUI", "TUI", "STN", "STY", "ATUI", "CVF"],
            usecols=["CUI", "STY"],
            nrows=1000 if debug else None,
        ).merge(mrconso[['CUI']])
        semantic_types = {CUI: sty_groups[STY] if use_sty_groups else STY for CUI, STY in zip(mrsty.CUI, mrsty.STY)}

        mrconso = mrconso[~(mrconso.STR.isna() | mrconso.CODE.isna())]
        mrconso['sab_index'] = mrconso['SAB'].apply(lambda x: preferred_sab.index(x) if x in preferred_sab else len(preferred_sab))
        mrconso['tty_index'] = mrconso['TTY'].apply(lambda x: preferred_tty.index(x) if x in preferred_tty else len(preferred_tty))
        mrconso['is_pref_index'] = (mrconso['ISPREF'] != 'Y').astype(int)
        mrconso['lang_index'] = mrconso['LAT'].apply(lambda x: preferred_lat.index(x) if x in preferred_lat else len(preferred_lat))
        mrconso = mrconso.sort_values(["CUI", "sab_index", "is_pref_index", "lang_index", "tty_index"])

        if try_improve_case:
            mrconso['STR'] = mrconso['STR'].apply(self.try_improve_case)

        mrconso = mrconso.drop_duplicates(["CUI", "STR"], keep="first")

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
