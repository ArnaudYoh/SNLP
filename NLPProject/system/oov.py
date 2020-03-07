import pickle
from utils import normalize, l2_nearest
import numpy as np


class OOV:
    """Class for Handling OOV words"""

    def __init__(self, lexicon, list_all_symbols, freq_tokens):

        self.lexicon = lexicon
        self.words_with_embeddings, self.embeddings = pickle.load(open('data/polyglot-fr.pkl', "rb"),
                                                                  encoding='bytes')  # or "bytes" or latin1"
        self.words_lexicon = list(freq_tokens.keys())

        # module for embedding similarity search (see below)
        self.Meaning_Search_Engine = Meaning_Search_Engine(self.words_lexicon, list_all_symbols,
                                                           self.words_with_embeddings, self.embeddings)

        # module for spelling correction (see below)
        self.Spelling_Corrector = Spelling_Corrector(self.words_lexicon,
                                                     self.words_with_embeddings,
                                                     freq_tokens)

    def closest_in_corpus(self, oov_word, viz_closest=False):

        # returns the closest word in the lexicon to the oov_word
        # (process combining embedding similarity and spelling correction
        # described in details in the report)
        # if viz_closest is True, print the steps done to find this closest token

        normalized = normalize(oov_word, self.Meaning_Search_Engine.word_lexicon_id)
        if not (normalized is None):
            return normalized

        if oov_word in self.words_with_embeddings:
            # look for words of corpus whose embedding is closest to the embedding of oov_word
            closest_corpus_word = self.Meaning_Search_Engine.closest_word_in_corpus(
                oov_word)  # , embeddings, words_lexicon
            if viz_closest: print(closest_corpus_word,
                                  " is the closest word (meaning) found among lexicon words having an embedding")
            return closest_corpus_word

        else:  # look for spelling errors
            correction = self.Spelling_Corrector.corrected_word(oov_word)

            if correction is None:
                if viz_closest: print("no corrected word (spelling) found at damerau-levenshtein distance less than 3")
                return None

            else:
                if viz_closest: print(correction,
                                      " is the closest word (spelling) found among words in the lexicon")  # or having an embedding

                if correction in self.words_lexicon:  # if corrected word in corpus
                    if viz_closest: print(correction, " is a word in the lexicon")
                    return correction

                else:  # if corrected word in embedding corpus
                    closest_corpus_word = self.Meaning_Search_Engine.closest_word_in_corpus(
                        correction)  # , embeddings, words_lexicon
                    if viz_closest: print(closest_corpus_word,
                                          " is the closest word (meaning) found among lexicon words having an embedding")
                    return closest_corpus_word


class Meaning_Search_Engine():

    # assuming the OOV word can be assigned an embedding,
    # this class enables to look for the closest neighbor in the lexicon (in term of cosine similarity between embeddings)

    def __init__(self, words_lexicon, list_all_symbols, words_with_embeddings, embeddings):

        self.list_all_symbols = list_all_symbols
        self.nb_all_symbols = len(list_all_symbols)

        self.words_lexicon = words_lexicon
        self.words_with_embeddings = words_with_embeddings
        self.embeddings = embeddings
        self.words_with_embeddings_id = {w: i for (i, w) in
                                         enumerate(self.words_with_embeddings)}  # Map words to indices

        self.build_embeddings_lexicon()  # precompute embeddings of words in lexicon
        self.embeddings_lexicon /= np.linalg.norm(self.embeddings_lexicon, axis=1)[:, None]  # normalize embeddings

    def build_embeddings_lexicon(self):
        # function computing the embeddings matrix of the words of lexicon having one : self.embeddings_lexicon
        # and the mapping id-word / word-id for these lexicon words : self.word_lexicon_id, self.id_word_lexicon

        self.embeddings_lexicon = None

        words_lexicon_in_corpus = []  # words of lexicon having an embedding

        # Embeddings of words of lexicon if possible
        for word in self.words_lexicon:
            word_normalized = normalize(word, self.words_with_embeddings_id)
            if not (word_normalized is None):
                words_lexicon_in_corpus.append(word)
                id_word = self.words_with_embeddings_id[word_normalized]
                if self.embeddings_lexicon is None:
                    self.embeddings_lexicon = self.embeddings[id_word]
                else:
                    self.embeddings_lexicon = np.vstack([self.embeddings_lexicon, self.embeddings[id_word]])

        # Map lexicon words present in embeddings to new ids and vice versa
        self.word_lexicon_id = {w: i for (i, w) in enumerate(words_lexicon_in_corpus)}
        self.id_word_lexicon = words_lexicon_in_corpus

    def closest_word_in_corpus(self, query):
        query = normalize(query, self.words_with_embeddings_id)
        if not query:
            print("OOV word")
            return None
        query_index = self.words_with_embeddings_id[query]
        query_embedding = self.embeddings[query_index]
        query_embedding /= np.linalg.norm(query_embedding)
        indices, distances = l2_nearest(self.embeddings_lexicon, query_embedding, 1)
        neighbors = [self.id_word_lexicon[idx] for idx in indices]
        return neighbors[0]


class Spelling_Corrector():

    ## Enables to find the closest neighbor in the lexicon of an OOV, in terms of spelling errors
    ## (lexicon is augmented with Polyglott embeddings if candidates_in_polyglott=True
    ##  in the end, I did not use this option, too expensive search)

    def __init__(self, words_lexicon, words_with_embeddings, freq_tokens):

        self.words_lexicon = words_lexicon  # we know frequencies of these words
        self.words_with_embeddings = words_with_embeddings  # we do not know frequencies of these words

        self.freq_tokens = freq_tokens

    def levenstein_damerau_distance(self, word, word2):
        # implementing levenstein distance computation
        # (damerau version to allow for swapping errors)

        size_word2 = len(word2)
        dist = np.zeros((3, size_word2 + 1))
        # dist[0,j] = distance from word[:t-1] to word2[:j-1]
        # dist[1,j] = distance from word[:t] to word2[:j-1]
        # dist[2,j] = distance from word[:t+1] to word2[:j-1]
        # where w[:-1] is the void string
        # and where t worths 0 at the beginning and progressively increases up to size_word1-1
        # (enables to reach a linear space complexity :
        # I do not save the whole matrix of distances between prefixes but only the last two lines)

        dist[0, :] = np.arange(size_word2 + 1)  # distance from void string to word2[:j]
        dist[1, 0] = 1
        for j in range(1, size_word2 + 1):
            diff_last_letters = word[0] != word2[j - 1]  # different last letters of prefixes
            dist[1, j] = min([dist[0][j] + 1, dist[1][j - 1] + 1, dist[0][j - 1] + diff_last_letters])

        for i in range(2, len(word) + 1):

            dist[2][0] = i  # distance from word[:i] to void string
            for j in range(1, size_word2 + 1):
                diff_last_letters = word[i - 1] != word2[j - 1]
                dist[2, j] = min([dist[1][j] + 1, dist[2][j - 1] + 1, dist[1][j - 1] + diff_last_letters])
                if j > 1:  # consider swap too !
                    if (word[i - 1] == word2[j - 2]) and (word[i - 2] == word2[j - 1]):
                        dist[2, j] = min(dist[2, j], dist[0, j - 2] + 1)

            dist[0, :] = dist[1, :]
            dist[1, :] = dist[2, :]

        return dist[2][size_word2]

    def corrected_word(self, query, candidates_in_polyglott=False):
        # finds the closest neighbor of query in the lexicon, in terms of spelling errors
        ## (lexicon is augmented with Polyglott embeddings if candidates_in_polyglott=True
        ##  in the end, I did not use this option, too expensive search)

        candidates = {1: [], 2: [], 3: []}  # words at distances 1,2,3 from real words
        min_dist = 3  # distance with closest word (if the closest is above 3, we return None)

        for word in self.words_lexicon:
            # we look for corrections in treebank corpus at most distance min_dist from query

            dist = self.levenstein_damerau_distance(query, word)
            if dist <= min_dist:
                candidates[dist].append(word)
                min_dist = dist

        if len(candidates[1]) > 0:
            # there is at least one word in treebank corpus at distance 1,
            # we return the most frequent of these

            idx_most_frequent = np.argmax([self.freq_tokens[word] for word in candidates[1]])
            return candidates[1][idx_most_frequent]

        #####################################
        if candidates_in_polyglott:  # too expensive in practice (we do not look for corrections in the Polyglott corpus=

            # if we reached this line
            # all words in treebank corpus are at distance more than 2
            # we look for corrections in embeddings corpus at most distance min_dist from query

            for word in self.words_with_embeddings:
                dist = self.levenstein_damerau_distance(query, word)
                if dist <= min_dist:
                    candidates[dist].append(word)
                    min_dist = dist
                if min_dist == 1:
                    # since no word at distance 1 was found previously in treebank corpus,
                    # we return the word which has an embedding and accomplished distance 1 with query
                    return candidates[1][0]

        #####################################
        # if we reached this line,
        # we found words in treebank/embeddings at at least distance 2 from the query

        list_candidates = candidates[min_dist]
        candidates_in_lexicon = []

        for word in list_candidates:
            if word in self.words_lexicon: candidates_in_lexicon.append(word)
        if len(candidates_in_lexicon) == 0:
            # the min distance is accomplished only by words which have an embedding
            # if any, we return one of these
            if len(list_candidates) == 0:
                return None
            return list_candidates[0]

        #####################################
        # if we reached this line
        # the min distace is accomplished by a word in treebank corpus, we return the most frequent of these
        idx_most_frequent = np.argmax([self.freq_tokens[word] for word in candidates_in_lexicon])
        return candidates_in_lexicon[idx_most_frequent]
