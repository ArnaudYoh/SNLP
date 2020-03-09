import argparse
from parser import Parser

# Read arguments
p = argparse.ArgumentParser()
p.add_argument('--test_file', type=str, required=False, help='Input file (text to parse)')
p.add_argument('--test_sentence', type=str, required=False, help='Input sentence (text to parse)')
args = p.parse_args()

train_ratio = 0.8
corpus = []
with open("data/sequoia-corpus+fct.mrg_strict", "r") as file_corpus:
    for line in file_corpus:
        corpus.append(line)

N = len(corpus)
nb_train = int(round(N * train_ratio))
corpus_train = corpus[:nb_train]

# Building Parser
print("Building PCFG and Parser")
my_parser = Parser(corpus_train)
print("Done")

print("Start Parsing")

if args.test_sentence:
    sent = args.test_sentence
    print("Sentence: ")
    print(sent + "\n")

    print("Parsing")
    my_parsing = my_parser.parse(sent)
    if my_parsing is None:
        print("Found no viable parsing.")
    else:
        print(my_parsing)

if args.test_file:
    counter = 0
    for sent in open(args.test_file):

        print("Sentence #{}:".format(counter))
        counter += 1
        print(sent + "\n")

        print("Parsing")
        my_parsing = my_parser.parse(sent)
        if my_parsing is None:
            print("Found no viable parsing.")
        else:
            print(my_parsing)
        print("\n\n")
