import re
import enchant


class Hashtag:
    def __init__(self, ner, urban, google):
        """Extracts features from a hashtag. These features are used in multi-task model.

        Parameters
        ----------
        ner : NamedEntityFeatures
            Stores Wikipedia titles. It is used to recognize named-entities.
        urban : UrbanDictFeatures
            Stores words from urban dictionary website.
        google : CountFeatures
            Google count features.
        """

        self.ner = ner
        self.urban_dict = urban
        self.google_counts = google
        self.dict = enchant.Dict("en_US")

        self.consonants = "qwrtypsdfghjklzxcvbnm"
        self.shape_xxddxx_right = re.compile("^([a-z]+)([0-9]+)$")
        self.shape_xxddxx_left = re.compile("^([0-9]+)([a-z]+)$")
        self.camel_case_re = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')

    def get_features(self, seg, best_cand):
        """Gets features from a hashtag.

        Parameters
        ----------
        seg : str
            Candidate segmentation.
        best_cand : str
            Best candidate segmentation according to beam search / word breaker.

        Returns
        -------
        List
            Feature vector for candidate segmentation.
        """

        target_org = "".join(seg.split())
        target = target_org.lower()

        features = list()
        # Length Features
        features.append(len(target))
        features.append((len(best_cand.split()) == 1) * 1.0)

        # Google ngram features
        features.append(self.google_counts.get_features(target))
        features.append(self.google_counts.get_features(best_cand.lower()))

        # Linguistic features
        features.append((self.dict.check(target)) * 1.0)
        features.append((target in self.ner.wiki_titles) * 1.0)
        features.append((target in self.ner.wiki_tokens) * 1.0)
        features.append((target in self.urban_dict.urban_words) * 1.0)

        # Word-shape features
        features.append((target.isdigit()) * 1.0)
        features.append(all([t in self.consonants for t in target]) * 1.0)

        features.append(1.0 if re.match(self.shape_xxddxx_left, target) else 0.0)
        features.append(1.0 if re.match(self.shape_xxddxx_right, target) else 0.0)
        features.append(1.0 * (len(re.findall(self.camel_case_re, target_org)) > 1 and
                               target not in self.ner.wiki_titles))

        return features
