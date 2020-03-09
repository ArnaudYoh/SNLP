import time

from utils import tagged_sent_to_tree, tree_to_sentence, tagged_to_sentence, save_scores
from PYEVALB import scorer
from PYEVALB import parser
from parser import Parser

# Get dataset
frac_train = 0.8
frac_dev = 0.1
frac_test = 0.1

corpus = []
with open("data/sequoia-corpus+fct.mrg_strict", "r") as file_corpus:
    for line in file_corpus:
        corpus.append(line)

N = len(corpus)
train_size = int(N * frac_train)
vali_size = int(N * frac_dev)
test_set = N - train_size - vali_size

dataset = dict()
dataset["train"] = corpus[:train_size]
dataset["dev"] = corpus[train_size:train_size + vali_size]
dataset["test"] = corpus[train_size + vali_size:]

# Build the Parser
print("Build CYK parser")
start = time.time()
my_parser = Parser(dataset["train"])
end = time.time()
print("Done in " + str(round(end - start, 2)) + "sec\n")

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

# Use pyevalb
if not len(sentences_test) == test_set or not len(real_parsings_test) == test_set:
    print("Issue with test set or train label creation")

for idx_sentence in range(test_set):

    real_parsing = real_parsings_test[idx_sentence]
    sent = sentences_test[idx_sentence]

    print("Sentence #" + str(idx_sentence))
    print(sent + "\n")

    print("Real Parsing")
    print(real_parsing + "\n")

    print("Our CYK Parsing")
    start = time.time()
    my_parsing = my_parser.parse(sent, viz_oov=False)
    if my_parsing is None:
        print("Found no viable parsing.")
    else:
        print(my_parsing)
    end = time.time()
    print("Done in " + str(round(end - start, 2)) + "sec\n\n\n")

    with open('results/evaluation_data.parser_output', 'a') as f:
        if my_parsing is None:
            f.write("Found no viable parsing." + "\n")
        else:
            f.write(my_parsing + "\n")

    if my_parsing is not None:
        # EVALPB works if we remove first and last brackets of the SEQUOIA format and the extra spaces that come with it 
        real_parsing = real_parsing[2:-1]
        my_parsing = my_parsing[2:-1]

        real_tree = parser.create_from_bracket_string(real_parsing)
        test_tree = parser.create_from_bracket_string(my_parsing)
        score = scorer.Scorer().score_trees(real_tree, test_tree)
        print('PYEVALB accuracy ' + str(score.tag_accracy))

        # for evaluation on the whole corpus, we save real_parsing 
        # and_my_parsing in new files without first and last brackets
        with open('results/real_parsings_test_for_eval.txt', 'a') as f:
            f.write(real_parsing + "\n")

        with open('results/my_parsings_test_for_eval.txt', 'a') as f:
            f.write(my_parsing + "\n")

save_scores('results/real_parsings_test_for_eval.txt',
            'results/my_parsings_test_for_eval.txt',
            'results/results_pyevalb.txt', )
