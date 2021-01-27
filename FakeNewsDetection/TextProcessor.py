import nltk
import html
import string
import re


class TextProcessor:
    """
    pre-processing text by removing stopwords, html entities and tags, tokenizing and eventually stemming tokens
    """

    all_stop_words = []
    stop_words = []
    all_words = []
    all_discriminative_words = []
    _stemmer = None

    def __init__(self, text, stemmer=None):
        self.all_stop_words = nltk.corpus.stopwords.words('english')                                    # initialize stop-word collection

        if stemmer is not None:
            self._stemmer = nltk.stem.snowball.SnowballStemmer(stemmer)                                 # initialize stemmer

        text = re.sub('<[^<]+?>', '', text)                                                             # removing html tags
        text = html.unescape(text)                                                                      # unescape html entities to be removed in stop-word elimination stage
        self.all_words = self.remove_punctuations(self.tokenizer(text))                                 # tokenizing the text and remove punctuation marks

        if self._stemmer is None:
            self.all_discriminative_words = self.remove_stopwords(self.all_words)                       # removing stop-words in token list --NO STEMMING of words
        else:
            self.all_discriminative_words = self.stemmer(self.remove_stopwords(self.all_words))         # removing stop-words in token list and get the stem of words
            self.stop_words = self.stemmer(self.stop_words)                                             # get the stem of all words in text

    def get_all_words(self, is_stemmed=True):
        if is_stemmed:
            return self.all_discriminative_words + self.stop_words
        else:
            return self.all_words

    def get_stop_words(self):
        return self.stop_words

    def get_words(self):
        return self.all_discriminative_words

    def stemmer(self, tokens):
        all_discriminative_stemmed_words = [self._stemmer.stem(w) for w in tokens]
        return all_discriminative_stemmed_words

    def remove_stopwords(self, tokens):
        all_discriminative_words = [w for w in tokens if w not in set(self.all_stop_words)]
        self.stop_words = [w for w in tokens if w in set(self.all_stop_words)]
        return all_discriminative_words

    @staticmethod
    def tokenizer(text):
        tokens = nltk.word_tokenize(text)
        tokens = [w.lower() for w in tokens]
        return tokens

    @staticmethod
    def remove_punctuations(tokens):
        """
        translation method is the fastest way to this but the most accurate method
        is to replace punctuation marks with space character... when searching for
        query words it will be hard and costly to search as substrings in doc words
        """
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        return words

    @staticmethod
    def word_stemmer(word, stemmer):
        s = nltk.stem.snowball.SnowballStemmer(stemmer)
        return s.stem(word)