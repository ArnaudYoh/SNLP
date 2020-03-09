from copy import deepcopy
from pcfg import PCFG
from oov import OOV

import numpy as np
from utils import tagged_sent_to_tree, tree_to_sentence


class CYK:
    """Class for applying the CYK algorithm"""

    def __init__(self, pcfg, oov):

        self.PCFG = pcfg
        self.OOV = oov

        self.tag_to_id = {tag: i for (i, tag) in enumerate(self.PCFG.list_all_tags)}

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

    def compute_cyk(self, sentence, viz_oov=False):
        """Apply the CYK algorithm (heavily influenced by https://en.wikipedia.org/wiki/CYK_algorithm)"""

        n = len(sentence)
        prob_matrix = np.zeros((n, n, self.PCFG.nb_all_tags))
        cyk_matrix = np.zeros((n, n, self.PCFG.nb_all_tags, 3))

        # probabilities of tags for unary rule
        for (position_word, word) in enumerate(sentence):

            token_to_tag = word

            if not (word in self.OOV.words_lexicon):
                token_to_tag = self.OOV.closest_in_corpus(word, verbose=viz_oov)

            if token_to_tag is None:
                for (tag, counts) in self.PCFG.freq_terminal_tags.items():
                    if tag in self.tag_to_id:
                        id_tag = self.tag_to_id[tag]
                        prob_matrix[position_word, 0, id_tag] = counts
            else:
                for (tag, proba) in self.PCFG.lexicon[token_to_tag].items():
                    if tag in self.tag_to_id:
                        id_tag = self.tag_to_id[tag]
                        prob_matrix[position_word, 0, id_tag] = proba

        for l in range(1, n):
            for s in range(n - l):
                for cut in range(0, l):
                    for idx_root_tag in self.grammar_dicts:
                        for idx_left_tag in self.grammar_dicts[idx_root_tag]:
                            proba_left_derivation = prob_matrix[s, cut, idx_left_tag]
                            if proba_left_derivation > prob_matrix[s, l, idx_root_tag]:  # save useless iterations

                                for idx_right_tag, proba_split in self.grammar_dicts[idx_root_tag][idx_left_tag].items():

                                    proba_right_derivation = prob_matrix[s + cut + 1, l - cut - 1, idx_right_tag]
                                    proba_decomposition = proba_split * proba_left_derivation * proba_right_derivation

                                    if proba_decomposition > prob_matrix[s, l, idx_root_tag]:
                                        prob_matrix[s, l, idx_root_tag] = proba_decomposition
                                        cyk_matrix[s, l, idx_root_tag] = [cut, idx_left_tag, idx_right_tag]

        self.prob_matrix = prob_matrix
        self.cyk_matrix = cyk_matrix.astype(int)
