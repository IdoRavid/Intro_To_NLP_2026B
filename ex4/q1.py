import random
import nltk
from nltk.corpus import dependency_treebank
from collections import defaultdict
from chu_liu_edmonds import cle_min

# Download the dependency treebank corpus if not already present
nltk.download('dependency_treebank', quiet=True)

# Load all parsed sentences
sentences = dependency_treebank.parsed_sents()

# Split into train/test: last 10% is test
split_idx = int(len(sentences) * 0.9)
train_set = sentences[:split_idx]
test_set = sentences[split_idx:]

print(f"Total sentences: {len(sentences)}")
print(f"Train set size:  {len(train_set)}")
print(f"Test set size:   {len(test_set)}")


# ---------------------------------------------------------------------------
# Feature function
# ---------------------------------------------------------------------------

def get_word(node):
    """Return the word form of a treebank node dict."""
    return node['word'] if node['word'] is not None else 'ROOT'


def get_tag(node):
    """Return the POS tag of a treebank node dict."""
    tag = node['tag']
    return tag if tag not in (None, 'TOP') else 'ROOT'


class FeatureMap:
    """
    Maintains the vocabulary of (word, word) and (tag, tag) pairs seen during
    training and maps them to integer indices in the sparse feature vector.
    """

    def __init__(self):
        self._index = {}   # (type, a, b) -> int
        self._frozen = False

    def _key_word(self, w, w2):
        return ('w', w, w2)

    def _key_tag(self, t, t2):
        return ('t', t, t2)

    def _get_or_add(self, key):
        if key not in self._index:
            if self._frozen:
                return None
            self._index[key] = len(self._index)
        return self._index[key]

    def freeze(self):
        """Stop adding new features (call after training vocabulary is built)."""
        self._frozen = True

    def feature_vector(self, u_node, v_node):
        """
        Return a sparse feature vector as a dict {index: 1} for the edge u->v.
        """
        w_idx = self._get_or_add(self._key_word(get_word(u_node), get_word(v_node)))
        t_idx = self._get_or_add(self._key_tag(get_tag(u_node), get_tag(v_node)))

        vec = {}
        if w_idx is not None:
            vec[w_idx] = 1
        if t_idx is not None:
            vec[t_idx] = 1
        return vec

    def __len__(self):
        return len(self._index)


def dot(weights, fvec):
    """Dot product between a weight dict/defaultdict and a sparse feature dict."""
    return sum(weights[i] * v for i, v in fvec.items())


# ---------------------------------------------------------------------------
# Gold tree helpers
# ---------------------------------------------------------------------------

def get_gold_edges(graph):
    """Return the set of (head, dependent) address pairs in the gold tree."""
    edges = set()
    for addr, node in graph.nodes.items():
        if addr == 0:
            continue
        if node['head'] is not None:
            edges.add((node['head'], addr))
    return edges


# ---------------------------------------------------------------------------
# MST decoder via Chu-Liu/Edmonds
# ---------------------------------------------------------------------------

def mst_decode(graph, weights, feature_map):
    """
    Return the highest-scoring dependency tree as a set of (head, dep) pairs.

    cle_min finds the MINIMUM arborescence, so we negate scores to get the max.
    """
    nodes = graph.nodes
    n = len(nodes)  # includes ROOT at address 0

    scores = {}
    for u_addr, u_node in nodes.items():
        for v_addr, v_node in nodes.items():
            if u_addr == v_addr or v_addr == 0:
                continue  # no self-loops, nothing points to ROOT
            fvec = feature_map.feature_vector(u_node, v_node)
            score = dot(weights, fvec)
            scores[(u_addr, v_addr)] = -score  # negate for min arborescence

    tree = cle_min(scores, n)  # {child: parent}
    return {(parent, child) for child, parent in tree.items()}


# ---------------------------------------------------------------------------
# Averaged structured perceptron
# ---------------------------------------------------------------------------

def train_perceptron(train_set, feature_map, n_iter=2):
    """
    Averaged structured perceptron for dependency parsing.
    Returns the averaged weight vector as a defaultdict(float).
    """
    weights = defaultdict(float)
    weights_sum = defaultdict(float)
    t = 0
    order = list(range(len(train_set)))

    for iteration in range(n_iter):
        random.shuffle(order)
        correct = 0

        for idx in order:
            graph = train_set[idx]
            gold_edges = get_gold_edges(graph)
            pred_edges = mst_decode(graph, weights, feature_map)

            if pred_edges != gold_edges:
                # w += f(gold) - f(pred)  (learning rate = 1)
                for (u, v) in gold_edges - pred_edges:
                    for feat_idx, val in feature_map.feature_vector(graph.nodes[u], graph.nodes[v]).items():
                        weights[feat_idx] += val
                for (u, v) in pred_edges - gold_edges:
                    for feat_idx, val in feature_map.feature_vector(graph.nodes[u], graph.nodes[v]).items():
                        weights[feat_idx] -= val
            else:
                correct += 1

            for k, v in weights.items():
                weights_sum[k] += v
            t += 1

        print(f"Iteration {iteration + 1}: {correct}/{len(order)} sentences correct")

    return defaultdict(float, {k: v / t for k, v in weights_sum.items()})


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

random.seed(123)

# Build feature vocabulary from training set
feature_map = FeatureMap()
for graph in train_set:
    nodes = graph.nodes
    for u_addr, u_node in nodes.items():
        for v_addr, v_node in nodes.items():
            if u_addr != v_addr and v_addr != 0:
                feature_map.feature_vector(u_node, v_node)
feature_map.freeze()
print(f"\nFeature vocabulary size: {len(feature_map)}")

weights = train_perceptron(train_set, feature_map, n_iter=2)


# ---------------------------------------------------------------------------
# Evaluation — Unlabeled Attachment Score (UAS)
# ---------------------------------------------------------------------------

def evaluate_uas(dataset, weights, feature_map):
    total_words = 0
    total_correct = 0

    for graph in dataset:
        gold_edges = get_gold_edges(graph)
        pred_edges = mst_decode(graph, weights, feature_map)
        n_words = len(graph.nodes) - 1  # exclude ROOT

        total_correct += len(gold_edges & pred_edges)
        total_words += n_words

    return total_correct / total_words


uas = evaluate_uas(test_set, weights, feature_map)
print(f"\nTest UAS: {uas:.4f}")
