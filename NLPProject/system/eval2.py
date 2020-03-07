from PYEVALB import scorer as PYEVALB_scorer

# evaluation on the whole corpus
PYEVALB_scorer.Scorer().evalb('results/real_parsings_test_for_eval.txt',
                              'results/my_parsings_test_for_eval.txt',
                              'results/results_pyevalb.txt',)