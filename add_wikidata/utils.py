from collections import defaultdict
from os.path import basename, dirname
from pickle import dump as pickle_dump
from pickle import load as pickle_load
from pickle import HIGHEST_PROTOCOL as pickle_HIGHEST_PROTOCOL
from requests import get as requests_get


# API_URL = 'api.conceptnet.io'
API_URL = 'localhost:8084'
PARTS_OF_SPEECH = ['n', 'v', 'r', 'a']


def save_obj(obj, name):
    with open(f'obj/{name}.pkl', 'wb') as f:
        pickle_dump(obj, f, pickle_HIGHEST_PROTOCOL)

def load_obj(name):
    with open(f'obj/{name}.pkl', 'rb') as f:
        return pickle_load(f)

def get_cn_url(cn_id):
    # return f'http://{API_URL}/{cn_id}?offset=0&limit=1000'
    return f'http://{API_URL}/query?node={cn_id}&other=/c/en&offset=0&limit=1000'


def get_cn_json(cn_id):
    url = get_cn_url(cn_id)
    return requests_get(url).json()


def get_root(cn_id):
    parts = cn_id.rsplit('/')
    root = parts[3]
    if len(parts) >= 5:
        part_of_speech = parts[4]
        if part_of_speech in PARTS_OF_SPEECH:
            return root + '/' + part_of_speech
    return root
    # last = basename(cn_id)
    # if last in ['n', 'v', 'r']:
    #     return basename(dirname(cn_id))
    # return last

def get_core(cn_id):
    parts = cn_id.rsplit('/')
    root = parts[3]
    return root

def get_standard_cn_id(cn_id):
    return f'/c/en/{get_root(cn_id)}'

def is_current(root_parent_composite, raw):
    root_raw_composite = get_root(raw)
    if '/' in root_parent_composite and '/' not in root_raw_composite:
        core_parent, part_of_speech_parent = root_parent_composite.rsplit('/')
        return core_parent == root_raw_composite

    if '/' not in root_parent_composite and '/' in root_raw_composite:
        core_raw, part_of_speech_raw = root_raw_composite.rsplit('/')
        return root_parent_composite == core_raw

    if '/' in root_parent_composite and '/' in root_raw_composite:
        core_parent, part_of_speech_parent = root_parent_composite.rsplit('/')
        core_raw, part_of_speech_raw = root_raw_composite.rsplit('/')
        return core_parent == core_raw and part_of_speech_parent == part_of_speech_raw

    if '/' not in root_parent_composite and '/' not in root_raw_composite:
        return root_parent_composite == root_raw_composite


def list_to_dict(edges_list):
    edges_dict = defaultdict(lambda: defaultdict(set))
    for e1, e2, r in edges_list:
        edges_dict[e1][e2].add(r)

    return edges_dict

def to_edges_tuples(list_of_dicts):
    return [(e['edge_start_standard'], e['edge_end_standard'], e['predicate']) for e in list_of_dicts]
