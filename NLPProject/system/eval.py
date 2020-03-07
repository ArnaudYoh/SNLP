import time

from utils import tagged_sent_to_tree, tree_to_sentence, tagged_to_sentence

from PYEVALB import scorer as PYEVALB_scorer
from PYEVALB import parser as PYEVALB_parser

from cyk import CYK

###########################################
# 1) SPLITTING COPUS IN TRAIN/DEV/TEST SET
# (NOTE: I did not make use of sentences of DEV set but provide possible improvements using these in the report)

corpus = []
with open("data/SEQUOIA_treebank", "r") as file_corpus:
    for line in file_corpus:
        corpus.append(line)

# Splitting corpus into train/dev/test set
frac_train = 0.8
frac_dev = 0.1
frac_test = 0.1

N = len(corpus)
nb_train = int(round(N * frac_train))
nb_dev = int(round(N * frac_dev))
nb_test = N - nb_train - nb_dev

dataset = dict()
dataset["train"] = corpus[:nb_train]
dataset["dev"] = corpus[nb_train:nb_train + nb_dev]
dataset["test"] = corpus[nb_train + nb_dev:]

###########################################
# 2) SAVING TEST SENTENCES IN A SEPARATED FILE

sentences_test = []
gold_parsings_test = []

for (idx_sentence, human_parsing) in enumerate(dataset["test"]):
    # removing functions in tags (parts after hyphen)
    # after their removal, we get our "gold parsing"
    T = tagged_sent_to_tree(human_parsing, remove_after_hyphen=True)
    gold_parsing = tree_to_sentence(T)
    gold_parsings_test.append(gold_parsing)

    sent = tagged_to_sentence(gold_parsing)
    sentences_test.append(sent)

with open('results/sentences_test.txt', 'w') as f:
    for item in sentences_test:
        f.write("%s\n" % item)

###########################################
# 3) BUILDING PARSER ON TRAIN CORPUS

# Building Parser
print("Build CYK parser")
tic = time.time()
my_CYK_parser = CYK(dataset["train"])
tac = time.time()
print("Done in " + str(round(tac - tic, 2)) + "s")

print("")
print("Some statistics")
nb_tags = len(my_CYK_parser.PCFG.list_all_symbols[:my_CYK_parser.PCFG.nb_tags])
nb_total = len(my_CYK_parser.PCFG.list_all_symbols)
nb_artificial = nb_total - nb_tags
nb_artificial_bin = len([r for r in my_CYK_parser.PCFG.list_all_symbols[my_CYK_parser.PCFG.nb_tags:] if "|" in r])
print(nb_tags, " grammar tags")
print(nb_artificial, " artificial symbols")
print(nb_artificial_bin, " artificial symbols from BIN rule")
print(nb_artificial - nb_artificial_bin, " artificial symbols from UNIT rule")
print(nb_total, " symbols (tags and artificial ones)")
print(len(my_CYK_parser.OOV.words_lexicon), " words in lexicon")
print(len(my_CYK_parser.OOV.Meaning_Search_Engine.id_word_lexicon), " words in lexicon having an embedding")
print(len(my_CYK_parser.OOV.words_with_embeddings), " words having an embedding")

###########################################
# 4) PARSING TEST SENTENCES AND ASSESSING WITH PYEVALB

assert (len(sentences_test) == nb_test)
assert (len(gold_parsings_test) == nb_test)

for idx_sentence in range(nb_test):

    print("##############################")

    gold_parsing = gold_parsings_test[idx_sentence]
    sent = sentences_test[idx_sentence]

    print("")
    print("Sentence N°" + str(idx_sentence))
    print("")
    print(sent)
    print("")

    print("Gold Parsing")
    print(gold_parsing)
    print("")

    print("My Parsing")
    tic = time.time()
    my_parsing = my_CYK_parser.parse(sent, viz_oov=False)
    if my_parsing is None:
        print("Found no parsing grammatically valid.")
    else:
        print(my_parsing)
    tac = time.time()
    print("Done in " + str(round(tac - tic, 2)) + "s")
    print("")

    with open('results/evaluation_data.parser_output', 'a') as f:
        if my_parsing is None:
            f.write("Found no parsing grammatically valid." + "\n")
        else:
            f.write(my_parsing + "\n")

    if not (my_parsing is None):
        # uncomment to draw the tree (requires pygraphviz library)
        # from draw_tree import draw_tree
        # draw_tree(my_parsing)

        gold_parsing = gold_parsing[2:-1]  # EVALPB works if we remove first and last brackets of the SEQUOIA format
        my_parsing = my_parsing[2:-1]  # EVALPB works if we remove first and last brackets of the SEQUOIA format

        print("Score PYEVALB :")
        gold_tree = PYEVALB_parser.create_from_bracket_string(gold_parsing)
        test_tree = PYEVALB_parser.create_from_bracket_string(my_parsing)
        result = PYEVALB_scorer.Scorer().score_trees(gold_tree, test_tree)
        print('Tagging accurracy ' + str(result.tag_accracy))

        # for evaluation on the whole corpus, we save gold_parsing and_my_parsing in new files without first & last brackets
        with open('results/gold_parsings_test_for_eval.txt', 'a') as f:
            f.write(gold_parsing + "\n")

        with open('results/my_parsings_test_for_eval.txt', 'a') as f:
            f.write(my_parsing + "\n")

# evaluation on the whole corpus
PYEVALB_scorer.Scorer().evalb('results/gold_parsings_test_for_eval.txt', 'results/my_parsings_test_for_eval.txt',
                              'results/results_pyevalb.txt')
