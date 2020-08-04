from unittest import TestCase
from nlstruct.environment import hash_object

from nlstruct.dataloaders import (
    load_ncbi_disease,
    load_medic_synonyms, load_alt_medic_mapping,
    load_bc5cdr,
    load_quaero, describe_quaero,
    load_n2c2_2019_task3,
    load_genia_events, load_genia_ner,
    load_i2b2_2012_temporal_relations
)
from nlstruct.environment.cache import nocache


class DataloadersTest(TestCase):
    def run(self, result=None):
        with nocache():
            super().run(result)

    def test_genia_events(self):
        genia_events = load_genia_events()
        self.assertEqual(str(genia_events), """Dataset(
  (docs):        1514 * ('doc_id', 'text', 'split')
  (mentions):   16232 * ('doc_id', 'mention_id', 'label', 'text')
  (fragments):  16232 * ('doc_id', 'mention_id', 'fragment_id', 'begin', 'end')
  (attributes):     0 * ('doc_id', 'mention_id', 'attribute_id', 'label', 'value')
  (relations):      0 * ('doc_id', 'relation_id', 'relation_label', 'from_mention_id', 'to_mention_id')
  (comments):       0 * ('doc_id', 'comment_id', 'mention_id', 'comment')
)""")
        self.assertEqual(hash_object(genia_events), """413ffd208f7fb1ba""")

    def test_genia_ner(self):
        genia_ner = load_genia_ner(version="3.02p", doc_attributes={"custom": "test"})
        self.assertEqual(str(genia_ner), """Dataset(
  (docs):        2000 * ('custom', 'doc_id', 'text', 'split')
  (mentions):   57096 * ('doc_id', 'mention_id', 'label', 'text')
  (fragments):  57096 * ('doc_id', 'mention_id', 'fragment_id', 'begin', 'end')
  (attributes):     0 * ('doc_id', 'mention_id', 'attribute_id', 'label', 'value')
)""")
        self.assertEqual(hash_object(genia_ner), """0117e66576a1bcaa""")

        genia_ner_raw_composite = load_genia_ner(version="3.02p", doc_attributes={"custom": "test"}, merge_composite_types=False)
        self.assertEqual(str(genia_ner_raw_composite), """Dataset(
  (docs):        2000 * ('custom', 'doc_id', 'text', 'split')
  (mentions):   56079 * ('doc_id', 'mention_id', 'label', 'text')
  (fragments):  56079 * ('doc_id', 'mention_id', 'fragment_id', 'begin', 'end')
  (attributes):     0 * ('doc_id', 'mention_id', 'attribute_id', 'label', 'value')
)""")
        self.assertEqual(hash_object(genia_ner_raw_composite), """71be8b9e24d8f884""")

    def test_ncbi_disease(self):
        dataset = load_ncbi_disease()
        self.assertEqual(str(dataset), """Dataset(
  (docs):       792 * ('doc_id', 'text', 'split')
  (mentions):  6881 * ('doc_id', 'mention_id', 'category', 'text')
  (labels):    7059 * ('label_id', 'doc_id', 'mention_id', 'label')
  (fragments): 6881 * ('doc_id', 'mention_id', 'begin', 'end', 'fragment_id')
)""")
        self.assertEqual(hash_object(dataset), """a6fb2086961da58b""")

    def test_bc5cdr(self):
        dataset = load_bc5cdr()
        self.assertEqual(str(dataset), """Dataset(
  (docs):       1501 * ('doc_id', 'text')
  (mentions):  28781 * ('doc_id', 'mention_id', 'category', 'text')
  (labels):    29072 * ('label_id', 'doc_id', 'mention_id', 'label')
  (fragments): 28781 * ('doc_id', 'mention_id', 'begin', 'end', 'fragment_id')
)""")
        self.assertEqual(hash_object(dataset), """adfb0e745d1142d6""")

    def test_n2c2_2019_task3(self):
        dataset = load_n2c2_2019_task3(validation_split=0.2, random_state=42)
        self.assertEqual(str(dataset), """Dataset(
  (docs):        100 * ('doc_id', 'text', 'split')
  (mentions):  13609 * ('doc_id', 'mention_id', 'label')
  (fragments): 13878 * ('doc_id', 'mention_id', 'fragment_id', 'begin', 'end')
)""")
        self.assertEqual(hash_object(dataset), """230e41163a3e040c""")

    def test_i2b2_2012_temporal_relations(self):
        dataset = load_i2b2_2012_temporal_relations(remove_empty_mentions=True, drop_duplicates=False)
        self.assertEqual(str(dataset), """Dataset(
  (docs):         310 * ('doc_id', 'split', 'text')
  (mentions):   34246 * ('doc_id', 'mention_id', 'label', 'text')
  (fragments):  34246 * ('doc_id', 'mention_id', 'fragment_id', 'begin', 'end')
  (relations):  60930 * ('doc_id', 'from_mention_id', 'label', 'relation_id', 'to_mention_id', 'mention_id_x', 'mention_id_y')
  (attributes): 64306 * ('attribute_id', 'doc_id', 'label', 'mention_id', 'value')
)""")
        self.assertEqual(hash_object(dataset), """c4b5aba42878b48b""")

    def test_quaero(self):
        dataset = load_quaero(version="2016", dev_split=0.2, seed=42)
        self.assertEqual(str(dataset), """Dataset(
  (docs):       2536 * ('doc_id', 'text', 'split', 'source')
  (fragments): 16377 * ('doc_id', 'mention_id', 'fragment_id', 'begin', 'end')
  (mentions):  16233 * ('doc_id', 'mention_id', 'label', 'text')
  (labels):    16283 * ('doc_id', 'mention_id', 'cui_id', 'cui')
)""")
        self.assertEqual(hash_object(dataset), """fcd8149e5f86b2ea""")
        self.assertEqual(str(describe_quaero(dataset)), """source           EMEA             MEDLINE            
split           train   dev  test   train   dev  test
files              11    12    15     833   832   833
labels            648   523   474    1860  1848  1908
mentions         2695  2260  2204    2994  2977  3103
unique_mentions   923   756   658    2296  2288  2390""")

    def test_medic_synonyms(self):
        self.assertEqual(hash_object(load_medic_synonyms("05-02-19")), """7f63a045c93c8264""")
        self.assertEqual(hash_object(load_alt_medic_mapping("05-02-19")), """5b37f761ea9b6db7""")
        self.assertEqual(hash_object(load_medic_synonyms("06-07-12")), """a1e31887294c4143""")
        self.assertEqual(hash_object(load_alt_medic_mapping("06-07-12")), """faaed71b73754ec1""")