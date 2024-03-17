# TODO: figure out decrementing hops and plus
# TODO: Correct None continue and is_relevant duplication logic
from random import sample
from pdb import set_trace as pdb_set_trace
from pprint import pformat, pprint
import warnings
from warnings import warn
from collections import defaultdict
from heapq import nlargest as heapq_nlargest
import gensim
import networkx as nx
from utils import get_cn_json, get_root, get_standard_cn_id, is_current, get_core
warnings.formatwarning = lambda msg, *args, **kwargs: f'{msg}\n'


ALLOWED_PREDICATES = ['/r/IsA', '/r/MannerOf', '/r/PartOf', '/r/RelatedTo', '/r/UsedFor'] # NOTE: remove synonym b/c not built into graph
CN_ID_ROOT = '/c/en'
SPLASH = \
'''
  _____   ____  _   _ ______
 |  __ \ / __ \| \ | |  ____|
 | |  | | |  | |  \| | |__
 | |  | | |  | | . ` |  __|
 | |__| | |__| | |\  | |____
 |_____/ \____/|_| \_|______|
'''


# Don't use lambda because it can't be pickled
def dd():
    return defaultdict(set)


def getter(x):
    return x[1]


MAPPER_ALLOWED_CHARACTERS = {
    '\'': '',
    '_': ''
}


class EdgeBuilder:
    def __init__(self, label_space_cn_ids):
        self.edge_list = []

        self.label_space_cn_ids = []
        for cn_id in label_space_cn_ids:
            standard = get_standard_cn_id(cn_id)

            self.label_space_cn_ids.append(standard)
        # self.visited = defaultdict(dd)
        self.visited_edge_ids = set()
        self.removed_nodes = defaultdict(set)
        # self.trees = {}
        self.graph = nx.MultiDiGraph()
        self.model = None


    def get_page_edges_objects(self, cn_id):
        '''Get the edges of the current cn_id'''
        response_json = get_cn_json(cn_id)
        edges = response_json['edges']
        # Navigate next pages
        while 'view' in response_json and 'nextPage' in response_json['view']:
            next_page = response_json['view']['nextPage']
            response_json = get_cn_json(next_page)
            edges.extend(response_json['edges'])

        return edges


    def find_match(self, new_edge):
        root_new_edge = get_root(new_edge)
        # if new edge root is strong (/v and /n), then if we only have a different strong then reject
        # if new edge root is weak is, then if we only have
        # TODO delete
        matched_ids = []
        for cn_id in self.label_space_cn_ids:
            if is_current(root_new_edge, cn_id):
                matched_ids.append(cn_id)
        if len(matched_ids) > 1:
            warn(f'find_match: nonunique matched_ids={matched_ids} for new_edge {new_edge}')
        return matched_ids[0] if len(matched_ids) > 0 else None


    def is_edge_visited(self, edge_obj):
        edge_start_matched = edge_obj['edge_start_matched']
        edge_end_matched = edge_obj['edge_end_matched']
        predicate = edge_obj['predicate']
        edge_id = edge_obj['edge_id']
        # def is_edge_id_visited(self, edge_start_standard, edge_end_standard, predicate, direction):
        # Standardize ID because
        # /a/[/r/IsA/,/c/en/built_in_bed/n/,/c/en/bed/n/] and
        # /a/[/r/IsA/,/c/en/built_in_bed/n/wn/artifact/,/c/en/bed/n/wn/artifact/]
        # have both been added because of strict string id checking

        # return edge_id in self.visited # this reduced edges from 72 to 46
        # # NEW: Sometimes the same exact edge_ids appear in multiple nodes, such as /c/en/against and /c/en/along
        # Some bidirectional edges have the same edge_id but others have reverse edge_ids
        # 1	/c/en/against	/c/en/with	/r/Antonym	/a/[/r/Antonym/,/c/en/against/,/c/en/with/]
        # 3	/c/en/against	/c/en/with	/r/Antonym	/a/[/r/Antonym/,/c/en/with/,/c/en/against/]
        # NOTE DECISION: use because semantic directionality is controlled by
        # the direction in the edge_id not the forward and back directions of
        # exploration. Otherwise there would be a new edge_id of the reverse
        # direction.
        return edge_id in self.visited_edge_ids or self.graph.has_edge(edge_start_matched, edge_end_matched, key=predicate)


    def visit_edge(self, edge_obj):
        # A node can have multiple predicates
        # TODO: node should have data['children'] = {(predicate,direction): edge_obj} pairs instead
        # of only one edge_obj across all (predicate, direction)
        edge_start_matched = edge_obj['edge_start_matched']
        edge_end_matched = edge_obj['edge_end_matched']
        predicate = edge_obj['predicate']
        edge_id = edge_obj['edge_id']
        level = edge_obj['level']
        # direction = edge_obj['direction']
        # NOTE: Decision: existing node is only interesting if it's in the
        # subtree of the starting node. After all, in a tree, multiple instances
        # of a concept/label can occur multiple times. This casts doubt on
        # the viability of the tree model. For the tree model to be working,
        # the nid for a node need to be prev_next_predicate, or the edge_id.

        # Instead of trees, it makes sense to have multigraph or a dict of graphs
        # with a predicate type each, like {'r/IsA': nx.Graph(), ...}

        self.graph.add_edge(edge_start_matched, edge_end_matched, key=predicate, predicate=predicate, edge_id=edge_id)
        print(f'edge #{self.graph.number_of_edges()}: level={level} {edge_start_matched} -> {predicate} -> {edge_end_matched}; edge_id={edge_id}')
        self.visited_edge_ids.add(edge_id)


    def get_raw_fact_from_response(self, response_json):
        return response_json['start']['@id'], \
               response_json['end']['@id'], \
               response_json['rel']['@id']


    def get_fact_from_response(self, response_json):
        cn_id_start = response_json['start']['@id']
        entity_1 = self.find_match(cn_id_start)
        cn_id_end = response_json['end']['@id']
        entity_2 = self.find_match(cn_id_end)
        entity_predicate = response_json['rel']['@id']
        return entity_1, entity_2, entity_predicate


    def run_dfs_relevant_plus(self, explore_cn_ids, hops=1, plus=0, strategy=None, limit=None, threshold=None, model=None, last_relevant=True):
        # hops is for how many levels of relevant hops
        # plus is for how many levels of hops regardless of relevance.
        if strategy == 'similarity':
            if model is None:
                self.model = gensim.models.KeyedVectors.load_word2vec_format('numberbatch-en-19.08.txt', binary=False)
            else:
                self.model = model

            if threshold is not None:
                self.threshold = threshold

        for cn_id in explore_cn_ids:
            cn_id_matched = self.find_match(cn_id)
            self.graph.add_node(cn_id_matched, level=0)
            visited_in_path = set()
            level = 0
            self.explore_cn_id(cn_id_matched, hops, plus, level, visited_in_path, strategy, limit, last_relevant)

        print(SPLASH)


    def explore_cn_id(self, cn_id, hops, plus, level, visited_in_path, strategy, limit, last_relevant):
        visited_in_path.add(cn_id)
        objs = self.get_page_edges_objects(cn_id)
        root_current = get_root(cn_id)
        core_current = get_core(cn_id)

        children_relevant = []
        children_irrelevant = []
        for idx, edge in enumerate(objs):
            e1_raw, e2_raw, r = self.get_raw_fact_from_response(edge)
            if r not in ALLOWED_PREDICATES:
                continue

            # Ditch non-English concepts for now
            if not e1_raw.startswith(CN_ID_ROOT) or not e2_raw.startswith(CN_ID_ROOT):
                continue

            # Exclude English braille
            # REVIEW: this might be too strict, like '/'
            if not e1_raw.isascii() or not e2_raw.isascii():
                continue

            e1_matched = self.find_match(e1_raw)
            e2_matched = self.find_match(e2_raw)
            is_relevant = bool(e1_matched) and bool(e2_matched)

            # Disallow dual-irrelevant edges if plus-hops <= 2
            if e1_matched == e2_matched and e2_matched is not None:
                warn(f'e1_matched=e2_matched={e2_matched} for cn_id = {cn_id} => edge {e1_raw, e2_raw, r}')
                continue


            # Decide to go down the start vs the end node in this edge
            # TODO: use matched instead in checks or assignment?
            edge_start_standard = get_standard_cn_id(e1_raw)
            edge_end_standard = get_standard_cn_id(e2_raw)

            edge_start_matched = e1_matched if e1_matched else edge_start_standard
            edge_end_matched = e2_matched if e2_matched else edge_end_standard

            if is_current(root_current, e1_raw):
                prev_node_raw = e1_raw
                prev_node_matched = e1_matched
                next_node_raw = e2_raw
                next_node_matched = e2_matched
                direction = 'forward'
            elif is_current(root_current, e2_raw):
                prev_node_raw = e2_raw
                prev_node_matched = e2_matched
                next_node_raw = e1_raw
                next_node_matched = e1_matched
                direction = 'back'
            else:
                pdb_set_trace()
                raise ValueError(f'run_dfs_relevant_plus: Error mismatch cn_id={cn_id}, e1_raw={e1_raw}, e2={e2_raw}')

            edge_id = edge['@id']
            next_node_raw_standard = get_standard_cn_id(next_node_raw)
            next_node_final = next_node_matched if next_node_matched else next_node_raw_standard

            if next_node_final in visited_in_path:
                continue

            # New mechanism: use lazy exploration: last must be relevant
            if last_relevant is True and hops == 0 and plus == 1:
                hops = 1
                plus = 0

            # 8 scenarios:
            # 1: relevant edge, no hops, no plus => x
            # 2: relevant edge, no hops, yes plus => √, decrement plus
            # 3: relevant edge, yes hops, no plus => √, decrement hops
            # 4: relevant edge, yes hops, yes plus => √, decrement hops
            # 5: irrelevant edge, no hops, no plus => x
            # 6: irrelevant edge, no hops, yes plus => √, decrement plus
            # 7: irrelevant edge, yes hops, no plus => x
            # 8: irrelevant edge, yes hops, yes plus => √, decrement hops
            # BUG: 10 hops 1 plus is the same as 1 hop 1 plus? Hops not working with plus=1
            if plus == 0 and (hops == 0 or not is_relevant):
                # Covers cases 1, 5, 7
                # if is_relevant:
                #     warn(f'no more hops and plus. Skipping relevant edge {edge_id} from cn_id={cn_id}')
                continue
            if hops == 0 and ((is_relevant and plus >= 1) \
                or (not is_relevant and plus >= 1)):
                # Covers cases 2, 6
                new_hops = hops
                new_plus = plus - 1
            elif hops >= 1 and (is_relevant \
                or (not is_relevant and plus >= 1)):
                # Covers cases 3, 4, and 8
                new_hops = hops - 1
                new_plus = plus
            else:
                raise ValueError(f'other cases encountered. is_relevant={is_relevant}, hops={new_hops}, plus={new_plus}')

            new_edge = {
                'idx': idx,
                'cn_id': cn_id,
                'edge_id': edge_id,
                'direction': direction,
                'edge_start_raw': e1_raw,
                'edge_start_matched': edge_start_matched,
                'edge_start_standard': edge_start_standard,
                'edge_end_raw': e2_raw,
                'edge_end_matched': edge_end_matched,
                'edge_end_standard': edge_end_standard,
                'predicate': r,
                'new_hops': new_hops,
                'new_plus': new_plus,
                'hops': hops,
                'plus': plus,
                'prev_node_raw': prev_node_raw,
                'prev_node_matched': prev_node_matched,
                'prev_node_raw_standard': get_standard_cn_id(prev_node_raw),
                'next_node_raw': next_node_raw,
                'next_node_matched': next_node_matched,
                'next_node_raw_standard': next_node_raw_standard,
                'next_node_final': next_node_final,
                'level': level,
            }
            if is_relevant:
                children_relevant.append(new_edge)
            else:
                children_irrelevant.append(new_edge)

        if strategy == 'random' and len(children_irrelevant) > limit:
            assert limit is not None
            # selected_children = sample(children, limit)
            selected_children_irrelevant = sample(children_irrelevant, limit)
        elif strategy == 'similarity' and len(children_irrelevant) > limit:
            assert limit is not None
            # TODO: need to preprocess cn_id and candidate so that only root is left
            candidate_similarities = []
            children_after_skip = []
            for child in children_irrelevant:
                key_next = get_core(child['next_node_raw'])
                try:
                    score = self.model.similarity(core_current, key_next) # get_standard_cn_id
                except KeyError as e:
                    warn(f'KeyError for {key_next} from cn_id={cn_id}. Message is {e}')
                    continue
                except:
                    pdb_set_trace()
                if self.threshold and score < self.threshold:
                    continue
                children_after_skip.append(child)
                candidate_similarities.append(score)

            largest_indices = heapq_nlargest(limit, enumerate(candidate_similarities), key=getter)
            selected_children_irrelevant = [children_after_skip[idx[0]] for idx in largest_indices]
        else:
            selected_children_irrelevant = children_irrelevant

        selected_children_combined = children_relevant + selected_children_irrelevant

        for child in selected_children_combined:
            edge_id = child['edge_id']

            edge_start_matched = child['edge_start_matched']
            edge_end_matched = child['edge_end_matched']
            direction = child['direction']
            level = child['level']

            next_node_raw = child['next_node_raw']
            next_node_matched = child['next_node_matched']
            next_node_raw_standard = child['next_node_raw_standard']
            next_node_final = child['next_node_final']

            prev_node_matched = child['prev_node_matched']
            prev_node_raw_standard = child['prev_node_raw_standard']
            prev_node = prev_node_matched if prev_node_matched else prev_node_raw_standard
            child['prev_node_final'] = prev_node
            if self.is_edge_visited(child):
                continue
            try:
                self.visit_edge(child)
            except Exception as e:
                pdb_set_trace()
                raise e

            cn_id = child['cn_id']
            new_hops = child['new_hops']
            new_plus = child['new_plus']
            hops = child['hops']
            plus = child['plus']
            idx = child['idx']

            self.explore_cn_id(next_node_final, new_hops, new_plus, level+1, visited_in_path, strategy, limit, last_relevant)


    def check_node_relevant(self, node):
        return bool(self.find_match(node.tag))


    def remove_node(self, tree, node):
        tree.remove_node(node.tag)
        self.removed_nodes[tree.root].add(node)
        # Remove by edge_id => remove by e1_standard, e2_standard, r
        node_child = node.data['child']
        edge_start_standard_node = node_child['edge_start_standard']
        edge_end_standard_node = node_child['edge_end_standard']
        predicate_node = node_child['predicate']
        for edge in self.edge_list:
            if edge_start_standard_node == edge['edge_start_standard'] \
                and edge_end_standard_node == edge['edge_end_standard'] \
                and predicate_node == edge['predicate']:
                self.edge_list.remove(edge)
                print(f'remove_node: deleted node {node.tag} with edge {edge_start_standard_node, edge_end_standard_node, predicate_node}')
                return
        pdb_set_trace()
        raise RuntimeError(f'remove_node: {node.tag} has not been deleted from self.edge_list')


    def is_node_removed(self, tree, node):
        node_tag = node.tag
        tree_root = tree.root
        if tree_root not in self.removed_nodes:
            return False
        for n in self.removed_nodes[tree_root]:
            if n.tag == node_tag:
                return True

        return False


    def get_leaf_nodes(self):
        graph = self.graph
        return [node for node in graph.nodes() if graph.in_degree(node)!=0 and graph.out_degree(node)==0]


    def prune(self):
        irrelevant_leaves = set(self.get_leaf_nodes()) - set(self.label_space_cn_ids)

        while irrelevant_leaves:
            self.graph.remove_nodes_from(irrelevant_leaves)
            irrelevant_leaves = set(self.get_leaf_nodes()) - set(self.label_space_cn_ids)
