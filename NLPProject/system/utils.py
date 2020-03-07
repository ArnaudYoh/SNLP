from copy import deepcopy

import numpy as np
from operator import itemgetter
import re
from PYEVALB import scorer
import networkx as nx


def tagged_to_sentence(tagged_sent):
    """Gets natural sentence from a tagged string"""
    tagged_sent = tagged_sent.split()  # into a list
    result = []
    for bloc in tagged_sent:
        if bloc[0] == "(":
            continue
        else:
            word = bloc.split(")")[0]
            result.append(word)
    return ' '.join(result)


def clean_tag(functional_tag):
    """Remove unnecessary tag information"""
    return functional_tag.split("-")[0]


def get_tag_list(grammar):
    """Returns list of all tags from a grammar"""
    res = set()
    for root_tag, rules in grammar.items():
        res.add(root_tag)
        for list_tags in rules:
            for tag in list_tags:
                res.add(tag)
    return list(res)


def increment_dict(d, word, tag, counts=1):
    """Increment the [word][tag] entry of d"""
    if type(tag) == list:
        tag = tuple(tag)
    if word in d.keys():
        d[word][tag] = d[word].get(tag, 0) + counts
    else:
        d[word] = {tag: counts}


def get_prob(d):
    """Converts counts to probabilities"""

    res = deepcopy(d)
    for (word, tags_counts) in d.items():
        total_counts = np.sum(list(tags_counts.values()))
        for tag in tags_counts.keys():
            res[word][tag] /= total_counts
    return res


def tagged_sent_to_tree(tagged_sent, remove_after_hyphen=True):
    """Returns a Tree from a tagged sentence"""
    max_id_node = 0

    tree = nx.DiGraph()

    sent = tagged_sent.split()
    hierarchy = list()

    hierarchy.append([])

    level = 0  # difference between the number of opened and closed parenthesis

    for (idx_bloc, bloc) in enumerate(sent):

        if bloc[0] == "(":

            if remove_after_hyphen:
                tag = clean_tag(bloc[1:])  # we add it to the hierarchy
            else:
                tag = bloc[1:]
            if level < len(hierarchy):  # there is already one tag as its level
                hierarchy[level].append((tag, max_id_node))
            else:  # first tag as its level
                hierarchy.append([(tag, max_id_node)])
            if idx_bloc > 0:
                tree.add_node(max_id_node, name=tag)
                max_id_node += 1
            level += 1

        else:

            word = ""
            nb_closing_brackets = 0
            for caract in bloc:
                if caract == ")":
                    nb_closing_brackets += 1
                else:
                    word += caract

            tree.add_node(max_id_node, name=word)
            tree.add_edge(max_id_node - 1, max_id_node)
            max_id_node += 1

            level -= nb_closing_brackets

            for k in range(nb_closing_brackets - 1, 0, -1):
                root = hierarchy[-2][-1][0]  # root tag
                id_root = hierarchy[-2][-1][1]
                if root == '':
                    break
                tags = hierarchy[-1]  # child tags

                for tag in tags:
                    tree.add_edge(id_root, tag[1])

                hierarchy.pop()

    return tree


def tree_to_sentence_helper(tree, node):
    """Partial sentence from subtree rooted at node"""
    children = list(tree.successors(node))
    if (len(children) == 1) and (len(list(tree.successors(children[0]))) == 0):
        return "(" + tree.nodes[node]["name"] + " " + tree.nodes[children[0]]["name"] + ")"
    else:
        res = "(" + tree.nodes[node]["name"]
        for child in sorted(children):
            res += " " + tree_to_sentence_helper(tree, child)
        res += ")"
        return res


def tree_to_sentence(tree):
    """Tree to sentence"""
    root = list(nx.topological_sort(tree))[0]
    return "( " + tree_to_sentence_helper(tree, root) + ")"


# The functions below are from https://nbviewer.jupyter.org/gist/aboSamoor/6046170

def case_normalizer(word, dictionary):
    """ In case the word is not available in the vocabulary,
     we can try multiple case normalizing procedure.
     We consider the best substitute to be the one with the lowest index,
     which is equivalent to the most frequent alternative."""
    w = word
    lower = (dictionary.get(w.lower(), 1e12), w.lower())
    upper = (dictionary.get(w.upper(), 1e12), w.upper())
    title = (dictionary.get(w.title(), 1e12), w.title())
    results = [lower, upper, title]
    results.sort()
    index, w = results[0]
    if index != 1e12:
        return w
    return word


DIGITS = re.compile("[0-9]", re.UNICODE)


def normalize(word, word_id):
    """ Find the closest alternative in case the word is OOV."""
    if not word in word_id:
        word = DIGITS.sub("#", word)
    if not word in word_id:
        word = case_normalizer(word, word_id)

    if not word in word_id:
        return None
    return word


def l2_nearest(embeddings, query_embedding, k):
    """Sorts words according to their Euclidean distance.
       To use cosine distance, embeddings has to be normalized so that their l2 norm is 1.
       indeed (a-b)^2"= a^2 + b^2 - 2a^b = 2*(1-cos(a,b)) of a and b are norm 1"""
    distances = (((embeddings - query_embedding) ** 2).sum(axis=1) ** 0.5)
    sorted_distances = sorted(enumerate(distances), key=itemgetter(1))
    return zip(*sorted_distances[:k])


def save_scores(in_file, truth_file, out_file="out.txt"):
    scorer.Scorer().evalb(in_file,
                          truth_file,
                          out_file)
