import re


class WordShapeFeatures:
    def __init__(self):
        """Extracts word-shape features.

        """

        self.shape_xxdd = re.compile("^([a-z]+)([0-9]+)$")
        self.shape_xxXX = re.compile("^([a-z]+)([A-Z]+)$")
        self.shape_ddxx = re.compile("^([0-9]+)([a-z]+)$")
        self.camel_case_re = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')

    def word_shape_xxdd(self, target, seg_tokens):
        """Gets number at the end word-shape feature.

        Parameters
        ----------
        target : str
            Hashtag.
        tokens : List
            Tokens in candidate segmentation.

        Returns
        -------
        Boolean
            Word-shape feature.
        """

        m = re.findall(self.shape_xxdd, target)
        if len(m) > 0 and len(m[0]) > 1 and len(seg_tokens) > 1:
            if m[0][1] == seg_tokens[1]:
                return 1.0
        return 0.0

    def word_shape_xxXX(self, target, seg_tokens):
        """Gets uppercase at the end word-shape feature.

        Parameters
        ----------
        target : str
            Hashtag.
        tokens : List
            Tokens in candidate segmentation.

        Returns
        -------
        Boolean
            Word-shape feature.
        """

        m = re.findall(self.shape_xxXX, target)
        if len(m) > 0 and len(m[0]) > 1 and len(seg_tokens) > 1:
            if m[0][1].lower() == seg_tokens[1].lower():
                return 1.0
        return 0.0

    def word_shape_ddxx(self, target, seg_tokens):
        """Gets number at the start word-shape feature.

        Parameters
        ----------
        target : str
            Hashtag.
        tokens : List
            Tokens in candidate segmentation.

        Returns
        -------
        Boolean
            Word-shape feature.
        """

        m = re.findall(self.shape_ddxx, target)
        if len(m) > 0 and len(m[0]) > 1 and len(seg_tokens) > 1:
            if m[0][0] == seg_tokens[0]:
                return 1.0
        return 0.0

    @staticmethod
    def word_shape_underscore(target, seg):
        """Gets underscore in the middle word-shape feature.

        Parameters
        ----------
        target : str
            Hashtag.
        seg : str
            Candidate segmentation.

        Returns
        -------
        Boolean
            Word-shape feature.
        """

        if "_" in target:
            return (seg == target.replace("_", " _ ").strip())*1.0
        return 0.0

    def word_shape_camel_case(self, target, seg):
        """Gets camel case word-shape feature.

        Parameters
        ----------
        target : str
            Hashtag.
        seg : str
            Candidate segmentation.

        Returns
        -------
        Boolean
            Word-shape feature.
        """

        matches = re.finditer(self.camel_case_re, target)
        seg_correct_camelcase = " ".join([m.group(0) for m in matches])
        return (seg_correct_camelcase == seg and target.lower() != target)*1.0

    @staticmethod
    def word_shape_digit(target, seg):
        """Gets number word-shape feature.

        Parameters
        ----------
        target : str
            Hashtag.
        seg : str
            Candidate segmentation.

        Returns
        -------
        Boolean
            Word-shape feature.
        """

        return (target.isdigit() and seg == target)*1.0

    @staticmethod
    def word_shape_abbr(target, seg):
        """Gets abbreviation word-shape feature.

        Parameters
        ----------
        target : str
            Hashtag.
        seg : str
            Candidate segmentation.

        Returns
        -------
        Boolean
            Word-shape feature.
        """
        return (all([t in "qwrtypsdfghjklzxcvbnm" for t in target]) and target == seg) * 1.0

    def get_features(self, seg, tokens):
        """Gets word-shape features.

        Parameters
        ----------
        seg : str
            Candidate segmentation.
        tokens : List
            Tokens in candidate segmentation.

        Returns
        -------
        Boolean List
            Word-shape features.
        """

        target = "".join(seg.split())
        tokens = [t.lower() for t in tokens]

        features = list()
        features.append(self.word_shape_xxdd(target.lower(), tokens))
        features.append(self.word_shape_ddxx(target.lower(), tokens))
        features.append(self.word_shape_camel_case(target, seg))
        features.append(self.word_shape_underscore(target, seg))
        features.append(self.word_shape_digit(target, seg))
        features.append(self.word_shape_abbr(target.lower(), seg.lower()))
        features.append(self.word_shape_xxXX(target, tokens))
        return features
