import re
from nltk.stem.porter import PorterStemmer


class NamedEntityFeatures:
    def __init__(self, wiki_file):
        """Extracts named-entity features.

        Parameters
        ----------
        wiki_file : str
            Path to wikipedia titles.
        """

        wiki_titles = set()
        wiki_tokens = set()
        for line in open(wiki_file):

            title = line.strip().lower()
            title = re.sub("[\(\[].*?[\)\]]", "", title)

            wiki_titles.add(title)
            wiki_tokens.update(title.split())

        self.wiki_titles = wiki_titles
        self.wiki_tokens = wiki_tokens
        self.stemmer = PorterStemmer()

    def get_features(self, seg, tokens):
        """Gets named-entity features. In other words, check if the segmentation is among Wikipedia titles.

        Parameters
        ----------
        seg : str
            Candidate segmentation.
        tokens : List
            Tokens in candidate segmentation.

        Returns
        -------
        Boolean List
            Named-entity features.
        """

        seg = seg.lower()
        title_flag = ((seg in self.wiki_titles) or
               (len(tokens) == 1 and self.stemmer.stem(seg) in self.wiki_titles))*1.0
        token_flag = ((seg in self.wiki_tokens) or
               (len(tokens) == 1 and self.stemmer.stem(seg) in self.wiki_tokens))*1.0
        return [title_flag, token_flag]
