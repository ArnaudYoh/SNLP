from copy import deepcopy

import numpy as np
from utils import get_tag_list, increment_dict, get_prob, clean_tag


class PCFG:
    """Class for forming the PCFG"""

    def __init__(self, corpus):

        self.grammar = dict()
        self.lexicon = dict()
        self.pre_terminal_tags = set()

        self.get_pcfg(corpus)

        self.lexicon_inverted = {tag: {} for tag in self.pre_terminal_tags}
        for word in self.lexicon:
            for tag in self.lexicon[word]:
                self.lexicon_inverted[tag][word] = self.lexicon[word][tag]

        self.freq_tokens = {}
        for word in self.lexicon.keys():
            for tag in self.lexicon[word].keys():
                if word in self.freq_tokens.keys():
                    self.freq_tokens[word] += self.lexicon[word][tag]
                else:
                    self.freq_tokens[word] = self.lexicon[word][tag]
        sum = np.sum(list(self.freq_tokens.values()))
        for word in self.freq_tokens:
            self.freq_tokens[word] /= sum

        self.set_artificial_tags = set()
        self.chomskyfy()

        self.freq_terminal_tags = {tag: np.sum(list(counts.values())) for (tag, counts) in self.lexicon.items()}
        sum = np.sum(list(self.freq_terminal_tags.values()))
        for tag in self.freq_terminal_tags:
            self.freq_terminal_tags[tag] /= sum

        self.grammar = get_prob(self.grammar)
        self.lexicon = get_prob(self.lexicon)

        list_all_tags = get_tag_list(self.grammar)
        self.list_artificial_symbols = list(self.set_artificial_tags)
        self.list_tags = list(set(list_all_tags).difference(self.set_artificial_tags))

        self.list_all_tags = self.list_tags + self.list_artificial_symbols
        self.nb_tags = len(self.list_tags)
        self.nb_all_tags = len(self.list_all_tags)

    def chomskyfy(self):
        """Apply the chomsky normalization"""

        self.binary_rule()
        self.unary_rule()

    def unary_rule(self):
        """Telescope unary rules"""
        copy_grammar = deepcopy(self.grammar)
        copy_lexicon = deepcopy(self.lexicon_inverted)

        rules_to_remove = []

        for A, rules in copy_grammar.items():
            for list_tags, counts in rules.items():
                if len(list_tags) == 1:  # unit rule A->B

                    B = list_tags[0]
                    rules_to_remove.append((A, list_tags))
                    freq = counts / (np.sum(list(self.grammar[A].values())))

                    # rule A -> B where B is a pre-terminal tag
                    if B in self.pre_terminal_tags and A != "SENT":
                        tag = A + "&" + B
                        self.set_artificial_tags.add(tag)

                        for word, counts2 in copy_lexicon[B].items():  # rule B -> word
                            # add A&B -> word, self.lexicon[word][A&B] = freq(A->B) * counts(B)
                            increment_dict(self.lexicon, word, tag, counts=counts2 * freq)

                        # for each rul X -> Y A, add rule X -> Y A&B
                        for root_tag2, rules2 in copy_grammar.items():
                            for list_tags2, counts2 in rules2.items():
                                if len(list_tags2) == 2 and list_tags2[1] == A:
                                    increment_dict(self.grammar, root_tag2, (list_tags2[0], tag), counts=counts2)

                    # If B not pre-terminal, for each rule B -> X Y, add A -> X Y
                    elif B not in self.pre_terminal_tags :
                        for list_tags_child, counts2 in copy_grammar[B].items():
                            if len(list_tags_child) == 2:
                                increment_dict(self.grammar, A, list_tags_child, counts=counts2 * freq)

        for (left, right) in rules_to_remove:
            del self.grammar[left][right]

    def binary_rule(self):
        """Replace all Rules with more than 2 children with a chain of rules"""
        copy_grammar = deepcopy(self.grammar)

        for A, rules in copy_grammar.items():
            for list_tags, counts in rules.items():
                nb_consecutive_tags = len(list_tags)

                if nb_consecutive_tags > 2:
                    del self.grammar[A][list_tags]

                    tag = A + "|" + '-'.join(list_tags[1:])
                    self.set_artificial_tags.add(tag)
                    increment_dict(self.grammar, A, (list_tags[0], tag), counts=counts)

                    for k in range(1, nb_consecutive_tags - 2):
                        new_tag = A + "|" + '-'.join(list_tags[k + 1:])
                        self.set_artificial_tags.add(new_tag)
                        increment_dict(self.grammar, tag, (list_tags[k], new_tag), counts=counts)
                        tag = new_tag

                    increment_dict(self.grammar, tag, (list_tags[-2], list_tags[-1]), counts=counts)

    def get_pcfg(self, corpus):
        """Get the PCFG"""

        for tagged_sent in corpus:

            sent = tagged_sent.split()
            levels = [[]]
            level = 0  # difference between the number of opened and closed parenthesis
            current_tag = None

            for bloc in sent:
                # Add a Tag
                if bloc[0] == "(":

                    tag = clean_tag(bloc[1:])
                    if level < len(levels):
                        levels[level].append(tag)
                    else:
                        levels.append([tag])

                    level += 1
                    current_tag = tag
                # Add a word
                else:
                    word = ""
                    nb_closing_parenthesis = 0
                    for caract in bloc:
                        if caract == ")":
                            nb_closing_parenthesis += 1
                        else:
                            word += caract
                    self.pre_terminal_tags.add(current_tag)
                    increment_dict(self.lexicon, word, current_tag)
                    level -= nb_closing_parenthesis

                    for _ in reversed(range(1, nb_closing_parenthesis)):
                        root = levels[-2][-1]
                        if root == '':
                            break
                        tags = levels[-1]
                        increment_dict(self.grammar, root, tags)
                        levels.pop()
