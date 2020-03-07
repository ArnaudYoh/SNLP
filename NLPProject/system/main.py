import argparse
from cyk import CYK

# Read arguments
p = argparse.ArgumentParser()
p.add_argument('--inFile', type=str, required=True, help='Input file (text to parse)')
p.add_argument('--outFile', type=str, required=False, default=None,
               help='Output file (will store parsings in bracketed format)')
p.add_argument('--vizOOV', type=bool, required=False, default=False, help='Plot management of OOV words')
args = p.parse_args()

corpus = []
with open("data/SEQUOIA_treebank", "r") as file_corpus:
    for line in file_corpus:
        corpus.append(line)

frac_train = 0.8
N = len(corpus)
nb_train = int(round(N * frac_train))
corpus_train = corpus[:nb_train]

# Building Parser
print("Building PCFG and Parser")
my_CYK_parser = CYK(corpus_train)
print("Done")

print("Start Parsing Text\n")

for sent in open(args.inFile):

    print("#################")
    print("Sentence : ")
    print(sent)
    print()

    print("Parsing")
    my_parsing = my_CYK_parser.parse(sent, viz_oov=args.vizOOV)
    if my_parsing is None:
        print("Found no parsing grammatically valid.")
    else:
        print(my_parsing)

    if not (args.outFile is None):
        with open(args.outFile, 'a') as f:
            if my_parsing is None:
                f.write("Found no parsing grammatically valid." + "\n")
            else:
                f.write(my_parsing + "\n")
