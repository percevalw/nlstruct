import pandas as pd

from nlstruct.environment.cache import cached
from nlstruct.environment.path import root
from nlstruct.utils.pandas import flatten


def get_raw_medic(version):
    file = root.resource("medic") / f"{version}.csv.gz"
    df = pd.read_csv(file, comment='#',
                     names=['DiseaseName', 'DiseaseID', 'AltDiseaseIDs', 'Definition', 'ParentIDs', 'TreeNumbers',
                            'ParentTreeNumbers', 'Synonyms', 'SlimMappings'])
    return df


@cached
def load_medic_synonyms(version, _cache=None):
    df = get_raw_medic(version)
    df['text'] = (df['DiseaseName']
                  .str.cat(df['Synonyms'].fillna(''), sep='|')
                  .str.strip('|').apply(lambda x: x.split('|')))
    df[['source', 'code']] = df['DiseaseID'].str.extract('(OMIM|MESH):(.*)')
    df['preferred'] = df['text'].apply(lambda x: tuple([True] + [False] * (len(x)-1)))
    df.loc[df['source'] == 'MESH', 'source'] = 'MSH'
    df['label'] = df['source'].str.cat(df['code'], sep=':')
    df = flatten(df[['text', 'label', 'code', 'source', 'preferred']])
    return df


def load_alt_medic_mapping(version, _cache=None):
    df = get_raw_medic(version)
    alt_medic_mapping = df[['AltDiseaseIDs', 'DiseaseID']].copy()
    alt_medic_mapping['AltDiseaseIDs'] = df['AltDiseaseIDs'].apply(
        lambda x: tuple(x.split('|')) if isinstance(x, str) else ())
    alt_medic_mapping = flatten(alt_medic_mapping, columns='AltDiseaseIDs')
    alt_medic_mapping = alt_medic_mapping.apply(lambda c: c.str.replace('MESH', 'MSH'))
    return dict(zip(alt_medic_mapping['AltDiseaseIDs'], alt_medic_mapping['DiseaseID']))
