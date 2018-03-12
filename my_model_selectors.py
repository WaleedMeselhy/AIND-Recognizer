import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        # TODO implement model selection based on BIC scores
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        min_bic = float('inf')
        best_model = None
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)
                logN = np.log(len(self.X))
                # N = sum(self.lengths)
                N, n_features = self.X.shape
                p = n_components ** 2 + 2 * n_features * n_components - 1

                        # calculate BIC score
                bic = -2 * logL + p * logN
                if bic < min_bic:
                    min_bic = bic
                    best_model = model
            except Exception as e:
                    continue
        return best_model if best_model else self.base_model(self.n_constant)



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def calc_logL_other_words(self, model, other_words):
        return [model[1].score(word[0], word[1]) for word in other_words]

    def calc_best_dic_score(self, score_dics):
    # Max of list of lists comparing each item by value at index 0
        return max(score_dics, key=lambda x: x[0])

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        other_words = []
        models = []
        dic_scores = []

        for word in self.words:
            if word != self.this_word:
                other_words.append(self.hwords[word])

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)
                models.append((logL, model))
            except Exception as e:
                pass
        for i, model in enumerate(models):
            logL_original_word, hmm_model = model
            score_dic = logL_original_word - np.mean(self.calc_logL_other_words(model, other_words))
            dic_scores.append(tuple([score_dic, model[1]]))
        return self.calc_best_dic_score(dic_scores)[1] if dic_scores else None



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        mean_scores = []

        # Save reference to 'KFold' in variable as shown in notebook
        split_method = KFold()
        try:
            for n_component in range(self.min_n_components, self.max_n_components + 1):
                model = self.base_model(n_component)
                # Fold and calculate model mean scores
                fold_scores = []
                for train_idx, test_idx in split_method.split(self.sequences):
                    self.X, self.lengths = combine_sequences(train_idx, self.sequences)
                    train_model = self.base_model(n_component)
                    X, lengths = combine_sequences(test_idx, self.sequences)
                    fold_scores.append(train_model.score(X, lengths))

                # Compute mean of all fold scores
                mean_scores.append(np.mean(fold_scores))
        except Exception as e:
            pass

        num_components = range(self.min_n_components, self.max_n_components + 1)
        states = num_components[np.argmax(mean_scores)] if mean_scores else self.n_constant
        return self.base_model(states)
