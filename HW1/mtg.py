import numpy as np

# to do:
# fix deterministic

class NGramModel:
    "Class for ngram model of size n"
    def __init__(self, n):
        """Generates an untrained n-gram model of size 1
        :param n: size of the n gram model
        :attribute self.params: Parameters for n_gram models of size 1-n, each in dictionary of format {token:
            (probability, earliest_location)}
        """
        self.n = n
        self.param_set = {_: {} for _ in range(1, self.n+1)}

    @staticmethod
    def update_params(params: dict, token: str, conditional:None):
        """
        Update some set up parameters by a token
        :param params: Existing vocab and counts
        :param token: New token to accommodate
        :param conditional: Set of previous tokens new token must follow (None for unigram)
        """

        # if no conditional
        if not conditional:
            if token in params.keys():
                params[token] += 1
            else:
                params[token] = 1

        else:
            cond_key = " ".join(conditional)
            if cond_key in params.keys():
                if token in params[cond_key].keys():
                    params[cond_key][token] += 1
                else:
                    params[cond_key][token] = 1
            else:
                params[cond_key] = {token: 1}
        return params


    def train(self, corpus: list):
        """Provided a corpus, iterate through it to build the parameter list in self.vocab
        :param corpus: training data in the form of a list
        """

        # For every token...
            # either add it to the vocab with count = 1
            # or if its there increment up the count
        for idx in range(len(corpus)):
            for grams in range(1, self.n+1):
                if (idx + grams - 1) <= len(corpus) -1:  # catch out of range idx ! this is not right
                    token = corpus[idx + grams - 1]  # -1 bc unigram model (n=1) should focus on current idx
                    if grams > 1:
                        conditional = corpus[idx: idx + grams - 1 ]
                    else:
                        conditional = None
                    params = self.param_set[grams]
                    params = self.update_params(params, token, conditional)
                    self.param_set[grams] = params

    def select_from_set(self, params: dict, key: str, randomize: bool):
        """Function for either selecting the most common value or by sampling from a distribution"""
        if randomize:
            if key:
                options = list(params[key].keys())
                weights = [list(params[key].values())[i] / sum(params[key].values()) for i in range(len(list(params[key].values())))]
            else:
                options = list(params.keys())
                weights = [list(params.value())[i] / sum(params.values()) for i in range(len(list(params.values())))]
            selection = np.random.choice(options, p=weights)
        else:
            if key:
                selection = max(params[key], key=params[key].get)
            else:
                selection = max(params, key=params.get)
        return selection


    def get_next(self, sentence: list, randomize: bool):
        """
        Given a sentence in list format, return the next expected word.
        :param sentence: Sentence that is to be continued
        :param randomize: if deterministic, select the most probable, otherwise sample
        :return:
        """
        # number of conditional tokens to use in model, n-1 bc a unigram has no conditionals
        num_cond_tokens = min(self.n - 1, len(sentence))

        pred = None
        while not pred:
            params = self.param_set[num_cond_tokens + 1]  # +1 bc set is 1 indexed, ex (bi gram, n=2, has 1 conditional)
            relevant_tokens = " ".join(sentence[-num_cond_tokens:])
            if num_cond_tokens > 0:
                if params.get(relevant_tokens): # if that value exists get returns
                    pred = self.select_from_set(params, relevant_tokens, randomize)
                else:
                    num_cond_tokens -= 1
            else:
                pred = self.select_from_set(params, None, randomize)
        return pred

def finish_sentence(sentence: list, n: int, corpus: list, randomize=False):
    """
    :param sentence: List of tokens to be built upon
    :param n: length of n-gram to use for prediction
    :param corpus: list of tokens used to train the n-gram model
    :param randomize: If true, select the most probable next token. If false, sample prom the appropriate distribution
    :return: completed sentence
    """
    # Init model based on provided parameters
    model = NGramModel(n)


    # train model based on corpus provided
    model.train(corpus)
    puncs = ['.', '?', '!']

    # produce new words (either random or not
    while len(sentence) < 10:

        next_token = model.get_next(sentence, randomize)
        sentence.append(next_token)

        if next_token in puncs:
            return sentence

    return sentence


