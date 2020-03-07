from copy import deepcopy

import numpy as np
from utils import get_tag_list, increment_dict, get_prob, clean_tag


class PCFG:
    """Class for forming the PCFG"""

    def __init__(self, corpus):

        self.grammar = dict()
        self.lexicon = dict()
        self.freq_tokens = dict()
        self.freq_terminal_tags = dict()

        self.list_artificial_tags = list()
        self.set_artificial_tags = set()
        self.list_tags = list()
        self.list_all_tags = list()

        self.nb_tags = 0
        self.nb_all_tags = 0

        self.build(corpus)

    def build(self, corpus):
        self.get_pcfg(corpus)

        # frequencies of each word/token
        for tag in self.lexicon.keys():
            for word in self.lexicon[tag].keys():
                if word in self.freq_tokens.keys():
                    self.freq_tokens[word] += self.lexicon[tag][word]
                else:
                    self.freq_tokens[word] = self.lexicon[tag][word]
        tot_count = np.sum(list(self.freq_tokens.values()))
        for word in self.freq_tokens:
            self.freq_tokens[word] /= tot_count

        self.chomskyfy()

        self.freq_terminal_tags = {tag: np.sum(list(counts.values())) for (tag, counts) in self.lexicon.items()}
        tot_count = np.sum(list(self.freq_terminal_tags.values()))
        for tag in self.freq_terminal_tags:
            self.freq_terminal_tags[tag] /= tot_count

        self.counts_to_prob()

        list_all_tags = get_tag_list(self.grammar)
        self.list_artificial_tags = list(self.set_artificial_tags)
        self.list_tags = list(set(list_all_tags).difference(self.set_artificial_tags))

        self.list_all_tags = self.list_tags + self.list_artificial_tags
        self.nb_tags = len(self.list_tags)
        self.nb_all_tags = len(self.list_all_tags)

    def counts_to_prob(self):
        """Go from counts to probabilities"""
        self.grammar = get_prob(self.grammar)
        self.lexicon = get_prob(self.lexicon)


    def chomskyfy(self):
        """Apply the chomsky normalization"""

        self.binary_rule()
        self.unary_rule()

    def unary_rule(self):

        copy_grammar = deepcopy(self.grammar)
        copy_lexicon = deepcopy(self.lexicon)

        rules_to_remove = []

        for A, rules in copy_grammar.items():
            for list_tags, counts in rules.items():
                if len(list_tags) == 1:  # unit rule A->B

                    B = list_tags[0]
                    rules_to_remove.append((A, list_tags))
                    freq = counts / (np.sum(list(self.grammar[A].values())))

                    # rule A -> B where B is a pre-terminal tag
                    if B in copy_lexicon:
                        if A != "SENT":
                            tag = A + "&" + B
                            self.set_artificial_tags.add(tag)

                            for word, counts2 in copy_lexicon[B].items():  # rule B -> word
                                # add A&B -> word, self.lexicon[word][A&B] = freq(A->B) * counts(B)
                                increment_dict(self.lexicon, tag, word,
                                               counts=counts2 * freq)

                            # for each rul X -> Y A, add rule X -> Y A&B
                            for root_tag2, rules2 in copy_grammar.items():
                                for list_tags2, counts2 in rules2.items():
                                    if len(list_tags2) == 2 and list_tags2[1] == A:
                                        increment_dict(self.grammar, root_tag2, (list_tags2[0], tag),
                                                       counts=counts2)

                    # If B not pre-terminal, for each rule B -> X Y, add A -> X Y
                    else:
                        for list_tags_child, counts2 in copy_grammar[B].items():
                            if len(list_tags_child) == 2:
                                increment_dict(self.grammar, A, list_tags_child,
                                               counts=counts2 * freq)

        for (left, right) in rules_to_remove:
            del self.grammar[left][right]

    def binary_rule(self):
        copy_grammar = deepcopy(self.grammar)

        for root_tag, rules in copy_grammar.items():
            for list_tags, counts in rules.items():
                nb_consecutive_tags = len(list_tags)

                if nb_consecutive_tags > 2:
                    del self.grammar[root_tag][list_tags]

                    tag = root_tag + "|" + '-'.join(list_tags[1:])
                    self.set_artificial_tags.add(tag)
                    increment_dict(self.grammar, root_tag, (list_tags[0], tag), counts=counts)

                    for k in range(1, nb_consecutive_tags - 2):
                        new_tag = root_tag + "|" + '-'.join(list_tags[k + 1:])
                        self.set_artificial_tags.add(new_tag)
                        increment_dict(self.grammar, tag, (list_tags[k], new_tag), counts=counts)
                        tag = new_tag

                    increment_dict(self.grammar, tag, (list_tags[-2], list_tags[-1]), counts=counts)

    def get_pcfg(self, corpus):
        """Get the PCFG"""

        for tagged_sent in corpus:

            sent = tagged_sent.split()
            levels = [[]]
            level = 0
            current_tag = None

            for part in sent:

                # Add Tag
                if part[0] == "(":
                    tag = clean_tag(part[1:])  # we add it to the hierarchy
                    if level < len(levels):  # there is already one tag as its level
                        levels[level].append(tag)
                    else:  # first tag as its level
                        levels.append([tag])

                    level += 1  # since we opened a new bracket
                    current_tag = tag  # saved in order to add the word to the lexicon

                # Add word
                else:
                    word = ""
                    nb_closing_brackets = 0
                    for caract in part:
                        if caract == ")":
                            nb_closing_brackets += 1
                        else:
                            word += caract
                    increment_dict(self.lexicon, current_tag, word)
                    level -= nb_closing_brackets

                    for _ in reversed(range(nb_closing_brackets)):
                        root = levels[-2][-1]
                        if root == '':
                            break
                        tags = levels[-1]  # children tags
                        increment_dict(self.grammar, root, tags)
                        levels.pop()


