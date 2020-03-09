import pickle
from utils import normalize, l2_nearest
import numpy as np


class OOV:
    """Class for Handling OOV words"""

    def __init__(self, lexicon, list_all_symbols, freq_tokens):

        self.lexicon = lexicon
        # Encoding should be "bytes" or latin1"
        self.words_with_embeddings, self.embeddings = pickle.load(open('data/polyglot-fr.pkl', "rb"), encoding='bytes')
        self.words_lexicon = list(freq_tokens.keys())
        self.list_all_symbols = list_all_symbols
        self.nb_all_symbols = len(list_all_symbols)

        self.word_frequency = freq_tokens

        self.words_with_embeddings_id = {w: i for (i, w) in
                                         enumerate(self.words_with_embeddings)}  # Map words to indices

        self.build_embeddings_lexicon()  # precompute embeddings of words in lexicon
        self.embeddings_lexicon /= np.linalg.norm(self.embeddings_lexicon, axis=1)[:, None]  # normalize embeddings

    def closest_in_corpus(self, oov_word, verbose=False):
        """Returns closest word using Levenshtein dist and cosine similarity of an embedding"""

        normalized = normalize(oov_word, self.word_lexicon_id)
        if normalized is not None:
            return normalized

        # OOV word exists in the embeddings
        if oov_word in self.words_with_embeddings:
            closest_corpus_word = self.closest_word_in_embedding(oov_word)
            if verbose:
                print(closest_corpus_word,
                      " is the closest word (meaning) found among lexicon words having an embedding")
            return closest_corpus_word

        # Look for closest and most frequent word with Levenshtein and then look into the embeddings
        else:
            correction = self.spell_corrected_word(oov_word)

            if correction is None:
                if verbose:
                    print("no word found at levenshtein distance less or equal to 2")
                return None

            else:
                if verbose:
                    print(correction, "closest word based on levenshtein dist")

                if correction in self.words_lexicon:  # if corrected word in corpus
                    if verbose:
                        print(correction, "is a word in the lexicon")
                    return correction

                else:
                    closest_corpus_word = self.closest_word_in_embedding(correction)
                    if verbose:
                        print(closest_corpus_word, "closest word based on embeddings")
                    return closest_corpus_word

    def build_embeddings_lexicon(self):
        """Builds embedding matrix and the word_idx mappings"""

        self.embeddings_lexicon = None

        words_lexicon_in_corpus = []  # words of lexicon having an embedding

        # Build Matrix
        for word in self.words_lexicon:
            word_normalized = normalize(word, self.words_with_embeddings_id)
            if word_normalized is not None:
                words_lexicon_in_corpus.append(word)
                id_word = self.words_with_embeddings_id[word_normalized]
                if self.embeddings_lexicon is None:
                    self.embeddings_lexicon = self.embeddings[id_word]
                else:
                    self.embeddings_lexicon = np.vstack([self.embeddings_lexicon, self.embeddings[id_word]])

        # Get Mappings
        self.id_word_lexicon = words_lexicon_in_corpus
        self.word_lexicon_id = {w: i for (i, w) in enumerate(words_lexicon_in_corpus)}

    def closest_word_in_embedding(self, query):
        """Returns word with the closest embedding to query"""
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

    def damerau_levenshtein_dist(self, w1, w2):
        """Computes the Damerau-Levenshtein distance"""

        n1 = len(w1)
        n2 = len(w2)
        dist = np.zeros((3, n2 + 1))  # Save space by not using n1 + 1 rows

        dist[0, :] = np.arange(n2 + 1)  # distance from void string to w2[:j]
        dist[1, 0] = 1


        for i in range(1, n2 + 1):
            diff_last_letters = 1 if w1[0] != w2[i - 1] else 0  # different last letters of prefixes
            dist[1, i] = min([dist[0][i] + 1, dist[1][i - 1] + 1, dist[0][i - 1] + diff_last_letters])

        for i in range(2, n1 + 1):
            dist[2, 0] = i  # distance from w1[:i] to void string

            for j in range(1, n2 + 1):
                diff_last_letters = 1 if w1[i - 1] != w2[j - 1] else 0
                dist[2, j] = min([dist[1][j] + 1, dist[2][j - 1] + 1, dist[1][j - 1] + diff_last_letters])
                if j > 1 and w1[i - 1] == w2[j - 2] and w1[i - 2] == w2[j - 1]:
                    dist[2, j] = min(dist[2, j], dist[0, j - 2] + 1)

            dist[0, :] = dist[1, :]  # Saving space
            dist[1, :] = dist[2, :]

        return dist[2][n2]

    def spell_corrected_word(self, query):
        """Returns closest word w.r.t. damerau-levenshtein dist with the highest frequency"""

        candidates = {1: [], 2: []}
        min_dist = max(candidates.keys())

        # Computes all distances
        for word in self.words_lexicon:
            dist = self.damerau_levenshtein_dist(query, word)
            if dist <= min_dist:
                candidates[dist].append(word)
                min_dist = dist

        list_candidates = candidates[min_dist]
        final_candidates = []

        for word in list_candidates:
            if word in self.words_lexicon:
                final_candidates.append(word)
        if len(final_candidates) == 0:
            if len(list_candidates) == 0:
                return None
            return list_candidates[0]

        idx_most_frequent = np.argmax([self.word_frequency[word] for word in final_candidates])
        return final_candidates[idx_most_frequent]