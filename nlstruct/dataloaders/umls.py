"""
Methods for frequent MySQL queries into UMLS.
"""
import contextlib
from copy import deepcopy

import pandas as pd
import regex
import tqdm.auto as tqdm

from nlstruct.environment.path import root
from nlstruct.environment.cache import cached

# from nlp.core.dask_client import progress

URL = 0
ID = 1

EXCLUDED_SOURCES = ('ICD10', 'eeee')

OBSOLETE_TYPES = '("IS", "LO", "MTH_IS", "MTH_LO", "MTH_OAF", "MTH_OAP", "MTH_OAS", "MTH_OET", "MTH_OF", "MTH_OPN", ' \
                 '"MTH_OP", "OAF", "OAM", "OAP", "OAS", "OA", "OET", "OF", "OLC", "OM", "ONP", "OOSN", "OPN", "OP")'

ATOM_REGEX = regex.compile('(.*[^\s])\s+\([^)]+\)\s*$')


# noinspection PyPep8Naming,SqlResolve
class UMLSMySQL:

    def __init__(self, host, user, password, database):
        import mysql.connector.pooling

        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.context_count = 0
        self.pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="umls_pool",
            pool_size=4,
            pool_reset_session=True,
            host=self.host, user=self.user,
            password=self.password, database=self.database)
        self.cursor = None

    @contextlib.contextmanager
    def get_cursor(self):
        conn = self.pool.get_connection()
        if self.cursor is None:
            self.cursor = conn.cursor()
        yield
        if self.cursor is not None:
            self.cursor.close()
            self.cursor = None
        conn.close()

    @staticmethod
    def close(conn, cursor):
        conn.close()
        cursor.close()

    def get_tree(self, cui, depth=0, exclude=()):
        result = (depth * 4) * " " + str(cui) + ' (' + ', '.join(self.get_UMLS_atoms(cui)) + ') \n'
        children = self.get_UMLS_children(cui, exclude=exclude)
        if children is None:
            return []
        descendants = deepcopy(children)

        i = 1
        for child in children:
            child_atoms = self.get_UMLS_atoms(child)
            if child not in exclude:
                (d, r) = self.get_tree(child, depth=depth + 1, exclude=exclude)
                descendants.update(d)
                result += r
            else:
                result += ((depth + 1) * 4) * " " + str(child) + ' (excluded) ({})\n'.format(
                    ", ".join(child_atoms))
            i += 1

        return descendants, result

    def get_UMLS_descendants(self, code, source='UMLS', depth=0, exclude=(), keep_path=None):
        if source != 'UMLS':
            cui = self.get_UMLS_cui_from_code(code, source)
        else:
            cui = code
        children = self.get_UMLS_children(cui, exclude=exclude, keep_path=keep_path)
        if children is None:
            return []
        descendants = deepcopy(children)

        i = 1
        for child in children:
            # print('   ' * depth + 'Parse {} ({} / {})'.format(child, i, len(children)))
            if child not in exclude:
                descendants.update(
                    self.get_UMLS_descendants(child, depth=depth + 1, exclude=exclude,
                                              keep_path=keep_path))
            i += 1

        return descendants

    def get_concepts_from_TUI(self, sty):
        query = f'select CUI from MRSTY WHERE TUI = "{sty}";'
        with self.get_cursor():
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
        return set([row[0] for row in rows])

    def get_concepts_from_STY(self, sty, only_sources=None):
        if only_sources is None or only_sources == []:
            query = f'select CUI from MRSTY WHERE STY = "{sty}";'
        else:
            sabs_str = ", ".join([f"'{s}'" for s in only_sources])
            query = f'select DISTINCT M1.CUI FROM MRSTY M1 JOIN MRCONSO M2 ON M1.cui = M2.cui ' \
                    f'AND M2.SUPPRESS = \'N\' ' \
                    f'WHERE M1.STY = "{sty}" AND M2.sab IN ({sabs_str});'
        with self.get_cursor():
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
        return set([row[0] for row in rows])

    def get_concepts_and_source_code_from_STY(self, sty, only_sources=None):
        if only_sources is None or only_sources == []:
            query = f'select DISTINCT M1.CUI, M2.CODE, M2.SAB ' \
                    f'FROM MRSTY M1 JOIN MRCONSO M2 ON M1.cui = M2.cui WHERE ' \
                    f'M1.STY = "{sty}" and SUPPRESS = \'N\';'
        else:
            sabs_str = ", ".join([f"'{s}'" for s in only_sources])
            query = f'select DISTINCT M1.CUI, M2.CODE, M2.SAB ' \
                    f'FROM MRSTY M1 JOIN MRCONSO M2 ON M1.cui = M2.cui ' \
                    f'WHERE M1.STY = "{sty}" AND M2.sab IN ({sabs_str}) and SUPPRESS = \'N\';'
        with self.get_cursor():
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
        return set([(row[0], row[1], row[2]) for row in rows])

    def get_UMLS_descendant_semantic_types(self, tui_list):
        ui3_str = ", ".join([f"'{s}'" for s in tui_list])
        query = f'SELECT b.ui FROM SRSTRE1 a, SRDEF b ' \
                f'WHERE ui3 IN ({ui3_str}) AND b.rt = "STY" AND a.ui1 = b.ui AND a.ui2 ' \
                f'IN (SELECT ui FROM SRDEF WHERE rt= "RL" and sty_rl = "isa") ORDER BY STN_RTN;'
        with self.get_cursor():
            self.cursor.execute(query)
            sty_rows = self.cursor.fetchall()
        return set([row[0].decode('utf-8') for row in sty_rows]) | set(tui_list)

    def get_UMLS_most_frequent_atom(self, cui, lang="ENG", remove_obsolete=True):
        if remove_obsolete:
            constraint = ' AND tty NOT IN ' + OBSOLETE_TYPES
        else:
            constraint = ''

        query = f'SELECT str FROM MRCONSO ' \
                f'WHERE cui = "{cui}" AND lat="{lang}" {constraint} and SUPPRESS = \'N\' ' \
                f'GROUP BY str ORDER BY COUNT(str) DESC, length(str) DESC LIMIT 1;'
        with self.get_cursor():
            self.cursor.execute(query)
            pref_atom = self.cursor.fetchone()
        if pref_atom is None:
            return ''
        else:
            return pref_atom[0]

    @staticmethod
    def rel2arrow(rel):
        if rel == 'RB':
            arrow = '<span class="rb" style="font-size: x-large; color: red;">&#8600;</span>'
        elif rel == 'RO':
            arrow = '<span class="ro" style="font-size: x-large; color: blue;">&rarr;</span>'
        else:
            raise ValueError(rel)
        return arrow

    def get_UMLS_related_concepts(self, cui, include_relations=(), exclude_relations=(),
                                  keep_path=None, exclude_cuis=(), exclude_stys=()):
        """
        Get all concepts related to specified concepts.

        Parameters
        ----------
        cui:
            the UMLS CUI
        include_relations
            the list of relations to consider. If empty, consider all relations but excluded relations
        exclude_relations
            the list of relations to exclude.
        exclude_stys
        exclude_cuis
        keep_path
        """
        include_clause = ''
        if len(include_relations) > 0:
            include_clause = ' AND rel IN (' + ', '.join(
                ["'" + rel + "'" for rel in include_relations]) + ')'
        exclude_rel_clause = ''
        if len(exclude_relations) > 0:
            exclude_rel_clause = ' AND rel NOT IN (' + ', '.join(
                ["'" + rel + "'" for rel in exclude_relations]) + ')'
        exclude_sty_clause = ''
        if len(exclude_stys) > 0:
            exclude_sty_clause = ' AND NOT EXISTS (SELECT * FROM MRSTY WHERE CUI=cui1 AND TUI IN ({}))'.format(
                ', '.join(["'" + tui + "'" for tui in exclude_stys]))

        clauses = include_clause + exclude_rel_clause + exclude_sty_clause
        query = f'SELECT cui1, rel FROM MRREL WHERE cui2 = "{cui}" AND sab="MTH" {clauses};'

        with self.get_cursor():
            self.cursor.execute(query)
            cui_rows = self.cursor.fetchall()

        if keep_path is None:
            children = set([row[0] for row in cui_rows if row[0] not in exclude_cuis])
        else:
            children = set([(row[0],
                             keep_path + self.rel2arrow(row[1]) + self.get_UMLS_most_frequent_atom(
                                 row[0])) for row in cui_rows if row[0] not in exclude_cuis])
            # children = set([(row[0], keep_path + '--' + self.rel2arrow(row[1]) + '->' + row[0]) for row in rows
            # if row[0] not in exclude])

        return children

    def get_UMLS_children(self, cui, exclude=(), keep_path=None):
        query = 'SELECT cui1 FROM MRREL WHERE cui2 = "{}" AND rel = "RB" AND sab="MTH";'.format(
            cui)
        with self.get_cursor():
            self.cursor.execute(query)
            rows = self.cursor.fetchall()

        if keep_path is None:
            children = set([row[0] for row in rows if row[0] not in exclude])
        else:
            children = set([(row[0], '>' + row[0]) for row in rows if row[0] not in exclude])

        return children

    def get_UMLS_cui_from_code(self, code, source):
        if source != 'UMLS':
            query = 'SELECT DISTINCT cui FROM MRCONSO WHERE SUPPRESS = \'N\' sab = "{}" AND code = "{}";'.format(
                source, code)
            with self.get_cursor():
                self.cursor.execute(query)
                rows = self.cursor.fetchall()
            return set([row[0] for row in rows])
        else:
            return code

    # def get_UMLS_cui_name(self, cui):
    #     # endpoint = '/content/current/CUI/{}'.format(cui)
    #     # concept_url = URI + endpoint
    #     # query = {'ticket':self.auth_client.getst(self.tgt)}
    #     # r = requests.get(concept_url,params=query)
    #     # r.encoding = 'utf-8'
    #     # items = json.loads(r.text)
    #     # return items["result"]["name"]
    #     return None

    def get_UMLS_atoms(self, atom_id, source='UMLS', lang="ENG", remove_obsolete=True):
        if remove_obsolete:
            constraint = 'AND M1.tty NOT IN ' + OBSOLETE_TYPES
        else:
            constraint = ''
        with self.get_cursor():
            if source != 'UMLS':
                query = f'SELECT DISTINCT M1.str FROM MRCONSO M1 JOIN MRCONSO M2 ON M1.cui = M2.cui ' \
                        f'WHERE M1.SUPPRESS = \'N\' AND M2.code = "{atom_id}" AND M2.sab = "{source}" AND M1.lat="{lang}" AND M1.sab ' \
                        f'NOT IN {EXCLUDED_SOURCES} {constraint};'
                self.cursor.execute(query)
            else:
                query = f'SELECT DISTINCT str FROM MRCONSO M1 ' \
                        f'WHERE M1.SUPPRESS = \'N\' AND cui = "{atom_id}" AND lat="{lang}" AND sab ' \
                        f'NOT IN {EXCLUDED_SOURCES} {constraint};'
                self.cursor.execute(query)
            rows = self.cursor.fetchall()
        atoms = set([row[0] for row in rows])
        return atoms

    @staticmethod
    def normalize_UMLS_atom(atom):
        m = ATOM_REGEX.match(atom)
        if m is not None:
            new_atom = m.group(1).rstrip()
        else:
            new_atom = atom
        return new_atom.lower()


UMLS_sty_groups = {
    'Anatomy': [
        'Anatomical Structure',
        'Body Location or Region',
        'Body Part, Organ or Organ Component',
        'Body Space or Junction',
        'Body Substance',
        'Body System',
        'Cell',
        'Cell Component',
        'Embryonic Structure',
        'Fully Formed Anatomical Structure',
        'Tissue'],
    'Biological Process or Function': [
        'Biologic Function',
        'Cell Function',
        'Genetic Function',
        'Molecular Function',
        'Natural Phenomenon or Process',
        'Organ or Tissue Function',
        'Organism',
        'Function',
        'Physiologic Function'],
    'Chemicals and drugs': [
        'Antibiotic',
        'Biomedical or Dental Material',
        'Carbohydrate Sequence',
        'Chemical',
        'Chemical Viewed Functionally',
        'Chemical Viewed Structurally',
        'Clinical Drug',
        'Hazardous or Poisonous Substance',
        'Inorganic Chemical',
        'Pharmacologic Substance',
        'Vitamin'],
    'Concept and Ideas': [
        'Classification',
        'Conceptual Entity',
        'Functional Concept',
        'Group Attribute',
        'Idea or Concept',
        'Intellectual Product',
        'Language',
        'Qualitative Concept',
        'Quantitative Concept',
        'Regulation or Law',
        'Spatial Concept'],
    'Devices': [
        'Drug Delivery Device',
        'Medical Device',
        'Research Device'],
    'Disorders': [
        'Acquired Abnormality',
        'Anatomical Abnormality',
        'Cell or Molecular Dysfunction',
        'Congenital Abnormality',
        'Disease or Syndrome',
        'Experimental Model of Disease',
        'Injury or Poisoning',
        'Mental or Behavioral Dysfunction',
        'Pathologic Function',
        'Neoplastic Process',
    ],
    'Genes and Proteins': [
        'Amino Acid, Peptide, or Protein',
        'Enzyme',
        'Lipid',
        'Immunologic Factor',
        'Indicator, Reagent, or Diagnostic Aid',
        'Gene or Genome',
        'Nucleic Acid, Nucleoside, or Nucleotide',
        'Receptor'],
    'Living Beings': [
        'Alga',
        'Amphibian',
        'Animal',
        'Archeon',
        'Bacterium',
        'Bird',
        'Fish',
        'Fungus',
        'Invertebrate',
        'Mammal',
        'Organism',
        'Plant',
        'Reptile',
        'Rickettsia or Chlamydia',
        'Vertebrate',
        'Virus'],
    'Medical procedures': [
        'Diagnostic Procedure',
        'Health Care Activity',
        'Laboratory Procedure',
        'Therapeutic or Preventive Procedure'],
    'Sign or Symptom': [
        'Sign or Symptom']
}
all_available_stys = [sty for sty_group in UMLS_sty_groups.values() for sty in sty_group]
UMLS_allowed_sources = {'RXNORM', 'MSH', 'SNOMEDCT_US', 'CHV', 'MDR', 'MEDCIN', 'OMIM'}


# noinspection PyTypeChecker
@cached.will_ignore(('db',))
def umls_synonyms_per_sty(sty,
                          umls_sources=('MSH', 'SNOMEDCT_US', 'CHV', 'MDR', 'MEDCIN'),
                          lang='ENG',
                          max_cuis=None,
                          db=None,
                          remove_obsolete=False,
                          _cache=None):
    """
    Extract synonyms in the UMLS database for a single semantic type

    Parameters
    ----------
    sty: str or list of str
    umls_sources: tuple of str
    lang: str
    max_cuis: None or int
    db: UMLSMySQL
    _cache: nlp.core.cache.CacheHandle

    Returns
    -------
    pd.DataFrame
    """
    if umls_sources is not None:
        # assert set(umls_sources) < UMLS_allowed_sources, "Unrecognized UMLS sources {}".format(
        #     ", ".join(list(set(umls_sources) - UMLS_allowed_sources)))
        pass

    if db is None:
        host, user, password, dbname = (
            root["UMLS_HOST"], root["UMLS_USER"], root["UMLS_PASSWORD"], root["UMLS_NAME"])
        db = UMLSMySQL(host, user, password, dbname)

    cuis = list(db.get_concepts_and_source_code_from_STY(sty, only_sources=umls_sources))
    desc = (sty[:17] + "..." if len(sty) > 20 else sty + " " * (20 - len(sty)))
    synonyms = []
    for cui, code, source in tqdm.tqdm(cuis[slice(0, max_cuis)], desc=desc):
        prefered = db.get_UMLS_most_frequent_atom(cui, lang=lang, remove_obsolete=False)
        if prefered:
            synonyms.append({"cui": cui, "code": code, "source": source, "text": prefered, "preferred": True})
            atoms = db.get_UMLS_atoms(cui, lang=lang, remove_obsolete=False)
            for atom in atoms:
                if atom != prefered:
                    synonyms.append({"cui": cui, "code": code, "source": source, "text": atom})
    if len(synonyms):
        synonyms.sort(key=lambda x: umls_sources.index(x["source"]))
        synonyms = pd.DataFrame(synonyms)
        synonyms['label'] = synonyms['source'].str.cat(synonyms['code'], sep=':')
        return synonyms
    return pd.DataFrame({"cui": [], "code": [], "source": [], "text": [], "preferred": []})


def load_umls_synonyms(
      sty,
      umls_sources=('MSH', 'SNOMEDCT_US', 'CHV', 'MDR', 'MEDCIN'),
      lang='ENG',
      max_cuis=None, **kwargs):
    """
    Extract synonyms in the UMLS database for one or many semantic types

    Parameters
    ----------
    sty: str or list of str
    umls_sources: list of str
    lang: str
    max_cuis: None or int

    Returns
    -------
    pd.DataFrame
    """
    sty_groups = [sty] if not isinstance(sty, (list, tuple)) else sty
    sty_list = [single_sty
                for item in sty_groups
                for single_sty in UMLS_sty_groups.get(item, [item])]
    # unrecognized_stys = [single_sty for single_sty in sty_list if single_sty not in all_available_stys]
    # assert len(unrecognized_stys) == 0, f"Unrecognized stys {', '.join(unrecognized_stys)}"

    res = []
    for single_sty in sty_list:
        sty_res = umls_synonyms_per_sty(single_sty, umls_sources, lang, max_cuis, **kwargs)
        sty_res['sty'] = single_sty
        res.append(sty_res)
    res = pd.concat(res, ignore_index=True, sort=False)
    # HACK because umls_synonyms_per_sty doesn't set preferred value for non prefered synonyms (should be false)
    res['preferred'] = res['preferred'].fillna(False)
    return res
