from nltk.stem.porter import PorterStemmer


class UrbanDictFeatures:
    def __init__(self, urban_file):
        """Extracts urban-dictionary features.

        Parameters
        ----------
        urban_file : str
            Path to Urban dictionary words.
        """

        urban_words = set()
        for line in open(urban_file):

            phrase = line.strip().lower()
            urban_words.add(phrase)

        self.urban_words = urban_words
        self.stemmer = PorterStemmer()

    def get_features(self, seg, tokens):
        """Gets urban-dictionary features. In other words, check if the segmentation is in Urban dictionary.

        Parameters
        ----------
        seg : str
            Candidate segmentation.
        tokens : List
            Tokens in candidate segmentation.

        Returns
        -------
        Boolean
            Urban dictionary features.
        """

        seg = seg.lower()
        if len(tokens) == 1:
            return ((seg in self.urban_words) or (self.stemmer.stem(seg) in self.urban_words))*1.0
        return (seg in self.urban_words)*1.0
