import time

from utils import tagged_sent_to_tree, tree_to_sentence, tagged_to_sentence, save_scores

from PYEVALB import scorer
from PYEVALB import parser

from cyk import CYK

# Get Train/Dev/Test
corpus = []
with open("data/sequoia-corpus+fct.mrg_strict", "r") as file_corpus:
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

# Save test in separate file
sentences_test = []
real_parsings_test = []

for (idx_sentence, human_parsing) in enumerate(dataset["test"]):
    T = tagged_sent_to_tree(human_parsing, remove_after_hyphen=True)
    real_parsing = tree_to_sentence(T)
    real_parsings_test.append(real_parsing)

    sent = tagged_to_sentence(real_parsing)
    sentences_test.append(sent)

with open('results/sentences_test.txt', 'w') as f:
    for item in sentences_test:
        f.write("%s\n" % item)

# Build CYK

print("Build CYK parser")
tic = time.time()
my_CYK_parser = CYK(dataset["train"])
tac = time.time()
print("Done in " + str(round(tac - tic, 2)) + "sec\n")


# Use pyevalb
assert (len(sentences_test) == nb_test)
assert (len(real_parsings_test) == nb_test)

for idx_sentence in range(nb_test):

    print("##############################")

    real_parsing = real_parsings_test[idx_sentence]
    sent = sentences_test[idx_sentence]

    print("Sentence #" + str(idx_sentence))
    print(sent + "\n")

    print("Real Parsing")
    print(real_parsing + "\n")

    print("Our CYK Parsing")
    tic = time.time()
    my_parsing = my_CYK_parser.parse(sent, viz_oov=False)
    if my_parsing is None:
        print("Found no viable parsing.")
    else:
        print(my_parsing)
    tac = time.time()
    print("Done in " + str(round(tac - tic, 2)) + "sec\n")

    with open('results/evaluation_data.parser_output', 'a') as f:
        if my_parsing is None:
            f.write("Found no viable parsing." + "\n")
        else:
            f.write(my_parsing + "\n")

    if my_parsing is not None:
        # EVALPB works if we remove first and last brackets of the SEQUOIA format and the extra spaces that come with it 
        real_parsing = real_parsing[2:-1]
        my_parsing = my_parsing[2:-1]

        print("Score PYEVALB:")
        real_tree = parser.create_from_bracket_string(real_parsing)
        test_tree = parser.create_from_bracket_string(my_parsing)
        result = scorer.Scorer().score_trees(real_tree, test_tree)
        print('accuracy ' + str(result.tag_accracy))

        # for evaluation on the whole corpus, we save real_parsing 
        # and_my_parsing in new files without first and last brackets
        with open('results/real_parsings_test_for_eval.txt', 'a') as f:
            f.write(real_parsing + "\n")

        with open('results/my_parsings_test_for_eval.txt', 'a') as f:
            f.write(my_parsing + "\n")

save_scores('results/real_parsings_test_for_eval.txt',
                              'results/my_parsings_test_for_eval.txt',
                              'results/results_pyevalb.txt',)
