## TP2 NLP project for the course on Algorithms for Speech and Natural Language Processing

In this project we implemented a parser which relies on the CYK algorithm and a PCFG grammar. 
The process is divided between several classes including a parser, a class for the pcfg, the cyk algorithm and a class for handling OOV words.

### How to run examples

If you want to quickly test the parser you can use our `run.sh` script or directly run our `main.py` file.

There are two options that you can use (you should use at least one option). For example:
```
./run.sh --test_sentence "Sentence that you want to parse"
```
or 
```
./run.sh --test_file "file_name of file containing one sentence per line"
```
