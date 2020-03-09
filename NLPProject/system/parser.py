import numpy as np

from cyk import CYK
from pcfg import PCFG
from oov import OOV
from utils import tagged_sent_to_tree, tree_to_sentence
from copy import deepcopy


class Parser:

    def __init__(self, corpus):
        self.PCFG = PCFG(corpus)
        self.OOV = OOV(self.PCFG.lexicon, self.PCFG.list_all_tags, self.PCFG.freq_tokens)
        self.CYK = CYK(self.PCFG, self.OOV)

        self.tag_to_id = {tag: i for (i, tag) in enumerate(self.PCFG.list_all_tags)}

    def parse(self, sentence, viz_oov=False):
        """Returns a parsed and tagged sentence from a natural sentence"""
        sentence = sentence.split()

        nb_words = len(sentence)

        if nb_words > 1:
            self.CYK.compute_cyk(sentence, viz_oov=viz_oov)
            idx_root_tag = self.tag_to_id["SENT"]
            if self.CYK.prob_matrix[0][nb_words - 1][idx_root_tag] == 0:  # no valid parsing
                return None
            parsing_list = self.parse_substring(0, nb_words - 1, idx_root_tag, sentence)

        else:
            word = sentence[0]
            token_to_tag = self.OOV.closest_in_corpus(word, verbose=viz_oov)
            if token_to_tag is None:
                tag = max(self.PCFG.freq_terminal_tags, key=self.PCFG.freq_terminal_tags.get)
            else:
                tag = max(self.PCFG.lexicon[token_to_tag], key=self.PCFG.lexicon[token_to_tag].get)
            parsing_list = "(" + tag + " " + word + ")"

        # converting the parsing stored as a string into a tree
        tree = tagged_sent_to_tree("( (SENT " + self.list_to_sentence(parsing_list) + "))",
                                   remove_after_hyphen=False)
        self.clean_tags(tree)
        return tree_to_sentence(tree)

    def list_to_sentence(self, parsing):
        """Go from list to string representation"""

        if type(parsing) == str:
            return parsing

        else:
            string = ""
            for p in parsing:
                root_tag = p[0]
                parsing_substring = p[1]
                string = string + "(" + root_tag + " " + self.list_to_sentence(parsing_substring) + ")" + " "
            string = string[:-1]  # Remove the extra space
            return string

    def clean_tags(self, tree):
        """Remove artificial tags and de-telescope tags"""

        # remove artificial tag of type X|X1X2X3.. (coming from BIN rule)
        nodes = deepcopy(tree.nodes)
        for node in nodes:
            children = list(tree.successors(node))
            if len(children) == 0:
                pass
            elif len(children) == 1 and len(list(tree.successors(children[0]))) == 0:
                pass
            else:
                father = list(tree.predecessors(node))
                if len(father) == 0:
                    pass
                else:
                    tag = tree.nodes[node]["name"]
                    if (self.tag_to_id[tag] >= self.PCFG.nb_tags) and (
                            "|" in tag):  # artificial tag from BIN rule
                        for child in tree.successors(node):
                            tree.add_edge(father[0], child)
                        tree.remove_node(node)

        # decomposing (A&B w) into (A (B w))
        max_id_node = np.max(tree.nodes())
        nodes = deepcopy(tree.nodes)
        for node in nodes:
            children = list(tree.successors(node))
            if len(children) == 0 or len(list(tree.predecessors(node))) == 0:
                pass
            elif len(children) == 1 and len(list(tree.successors(children[0]))) == 0:
                tag = tree.nodes[node]["name"]

                if (self.tag_to_id[tag] >= self.PCFG.nb_tags) and (
                        "&" in tag):  # artificial tag from UNIT rule
                    word = children[0]

                    idx_cut = None
                    for (idx, c) in enumerate(tag):
                        if c == "&":
                            idx_cut = idx

                    tree.nodes[node]["name"] = tag[:idx_cut]

                    idx_pre_terminal_node = max_id_node + 1
                    tree.add_node(idx_pre_terminal_node, name=tag[idx_cut + 1:])
                    max_id_node += 1

                    tree.remove_edge(node, word)
                    tree.add_edge(node, idx_pre_terminal_node)
                    tree.add_edge(idx_pre_terminal_node, word)

    def parse_substring(self, start, length, idx_root_tag, sentence):
        """Parse part of a sentence into a list"""

        if length == 0:
            return sentence[start]

        else:  # split enabling to reach max_proba_derivation[s,l,idx_root_tag]
            cut = self.CYK.cyk_matrix[start, length, idx_root_tag, 0]
            idx_left_tag = self.CYK.cyk_matrix[start, length, idx_root_tag, 1]
            idx_right_tag = self.CYK.cyk_matrix[start, length, idx_root_tag, 2]

            left_tag = self.PCFG.list_all_tags[idx_left_tag]
            right_tag = self.PCFG.list_all_tags[idx_right_tag]

            return [[left_tag, self.parse_substring(start, cut, idx_left_tag, sentence)],
                    [right_tag, self.parse_substring(start + cut + 1, length - cut - 1, idx_right_tag, sentence)]]
