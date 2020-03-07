from copy import deepcopy
from pcfg import PCFG
from oov import OOV

import numpy as np
from utils import tagged_sent_to_tree, tree_to_sentence


class CYK:
    """Class for applying the CYK algorithm"""

    def __init__(self, corpus_train):

        self.PCFG = PCFG(corpus_train)
        self.OOV = OOV(self.PCFG.lexicon, self.PCFG.list_all_tags, self.PCFG.freq_tokens)

        self.tag_to_id = {tag: i for (i, tag) in enumerate(self.PCFG.list_all_tags)}

        self.lexicon_inverted = {word: {} for word in self.OOV.words_lexicon}
        for tag in self.PCFG.lexicon:
            for word in self.PCFG.lexicon[tag]:
                self.lexicon_inverted[word][tag] = self.PCFG.lexicon[tag][word]

        # self.grammar_dicts[X][Y][Z] stores P(rule X->YZ)
        self.grammar_dicts = {}
        for (root_tag, rules) in self.PCFG.grammar.items():
            # root_tag is the left hand tag of the grammar rule
            idx_root_tag = self.tag_to_id[root_tag]
            self.grammar_dicts[idx_root_tag] = {}
            dico = {}
            for (split, proba) in rules.items():  # split is the right hand term, and proba the probability of the rule
                idx_left_tag = self.tag_to_id[split[0]]
                idx_right_tag = self.tag_to_id[split[1]]
                if idx_left_tag in dico.keys():
                    dico[idx_left_tag][idx_right_tag] = proba
                else:
                    dico[idx_left_tag] = {idx_right_tag: proba}
            self.grammar_dicts[idx_root_tag] = dico

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

    def parse_substring(self, s, l, idx_root_tag, sentence):
        """Parse part of a sentence into a list"""

        if l == 0:
            return sentence[s]

        else:  # split enabling to reach max_proba_derivation[s,l,idx_root_tag]
            cut = self.cyk_matrix[s, l, idx_root_tag, 0]
            idx_left_tag = self.cyk_matrix[s, l, idx_root_tag, 1]
            idx_right_tag = self.cyk_matrix[s, l, idx_root_tag, 2]

            left_tag = self.PCFG.list_all_tags[idx_left_tag]
            right_tag = self.PCFG.list_all_tags[idx_right_tag]

            return [[left_tag, self.parse_substring(s, cut, idx_left_tag, sentence)],
                    [right_tag, self.parse_substring(s + cut + 1, l - cut - 1, idx_right_tag, sentence)]]

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

    def compute_CYK(self, sentence, viz_oov=False):
        """Apply the CYK algorithm (heavily influenced by https://en.wikipedia.org/wiki/CYK_algorithm)"""

        n = len(sentence)
        prob_matrix = np.zeros((n, n, self.PCFG.nb_all_tags))
        cyk_matrix = np.zeros((n, n, self.PCFG.nb_all_tags, 3))

        # probabilities of tags for unary rule
        for (position_word, word) in enumerate(sentence):

            token_to_tag = word

            if not (word in self.OOV.words_lexicon):
                token_to_tag = self.OOV.closest_in_corpus(word, viz_closest=viz_oov)

            if token_to_tag is None:
                for (tag, counts) in self.PCFG.freq_terminal_tags.items():
                    if tag in self.tag_to_id:
                        id_tag = self.tag_to_id[tag]
                        prob_matrix[position_word, 0, id_tag] = counts
            else:
                for (tag, proba) in self.lexicon_inverted[token_to_tag].items():
                    if tag in self.tag_to_id:
                        id_tag = self.tag_to_id[tag]
                        prob_matrix[position_word, 0, id_tag] = proba

        for l in range(1, n):
            for s in range(n - l):
                for idx_root_tag in self.grammar_dicts:
                    for cut in range(0, l):
                        for idx_left_tag in self.grammar_dicts[idx_root_tag]:
                            proba_left_derivation = prob_matrix[s, cut, idx_left_tag]
                            if proba_left_derivation > prob_matrix[s, l, idx_root_tag]:  # save useless iterations

                                for (idx_right_tag, proba_split) in self.grammar_dicts[idx_root_tag][
                                    idx_left_tag].items():

                                    proba_right_derivation = prob_matrix[s + cut + 1, l - cut - 1, idx_right_tag]
                                    proba_decomposition = proba_split * proba_left_derivation * proba_right_derivation

                                    if proba_decomposition > prob_matrix[s, l, idx_root_tag]:
                                        prob_matrix[s, l, idx_root_tag] = proba_decomposition
                                        cyk_matrix[s, l, idx_root_tag] = [cut, idx_left_tag, idx_right_tag]

        self.prob_matrix = prob_matrix
        self.cyk_matrix = cyk_matrix.astype(int)

    def parse(self, sentence, viz_oov=False):
        """Returns a parsed and tagged sentence from a natural sentence"""
        sentence = sentence.split()

        nb_words = len(sentence)

        if nb_words > 1:
            self.compute_CYK(sentence, viz_oov=viz_oov)
            idx_root_tag = self.tag_to_id["SENT"]
            if self.prob_matrix[0][nb_words - 1][idx_root_tag] == 0:  # no valid parsing
                return None
            parsing_list = self.parse_substring(0, nb_words - 1, idx_root_tag, sentence)

        else:
            word = sentence[0]
            token_to_tag = self.OOV.closest_in_corpus(word, viz_closest=viz_oov)
            if token_to_tag is None:
                tag = max(self.PCFG.freq_terminal_tags, key=self.PCFG.freq_terminal_tags.get)
            else:
                tag = max(self.lexicon_inverted[token_to_tag], key=self.lexicon_inverted[token_to_tag].get)
            parsing_list = "(" + tag + " " + word + ")"

        # converting the parsing stored as a string into a tree
        tree = tagged_sent_to_tree("( (SENT " + self.list_to_sentence(parsing_list) + "))",
                                   remove_after_hyphen=False)
        self.clean_tags(tree)
        return tree_to_sentence(tree)
