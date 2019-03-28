import re
import enchant
import editdistance
from features.lm import LMFeatures
from features.hashtag import Hashtag
from features.counts import CountFeatures
from features.wordshapes import WordShapeFeatures
from features.urban_dict import UrbanDictFeatures
from features.named_entity import NamedEntityFeatures


class FeatureExtractor:
    def __init__(self, resources, model):
        """Extracts features from each candidate segmentation.

        Parameters
        ----------
        resources : Dictionary
            Paths to language model, ner, urban-dictionary, google counts and twitter count files.
        model: str
           Type of model. The model can be mse, mr, mse + multitask or mr + multitask. Please look at the paper for
           details.

        """

        print("Loading resources")
        self.model = model
        self.dict = enchant.Dict("en_US")

        self.ws = WordShapeFeatures()
        self.lm_gt = LMFeatures(resources["lm_gt"])
        self.lm_kn = LMFeatures(resources["lm_kn"])
        self.ner = NamedEntityFeatures(resources["wiki"])
        self.urban_dict = UrbanDictFeatures(resources["urban"])
        self.google_counts = CountFeatures(resources["google"])
        self.twitter_counts = CountFeatures(resources["twitter"])

        self.hashtag = Hashtag(self.ner, self.urban_dict, self.google_counts)

        print("Done loading resources")

    def get_features(self, input, topk):
        """Gets features from all candidates in a file.

        Parameters
        ----------
        file_name : str
            Input file name.

        Returns
        ------
        List
            List of feature vectors. Position i will have a list of feature vectors for candidates of hashtag i.
        List
            List edit distance values with respect to gold truth.
        List
            List of all candidate segmentations
        List
            List of all gold-truth segmentations.
        """

        print("Extracting features")
        all_candidates = {}
        for line in open(topk):
            tokens = line.strip().split("\t")
            all_candidates[tokens[0]] = tokens[1:]

        all_features, all_segs, all_gold, all_labels = [], [], [], []
        for line in open(input):

            tokens = line.strip().split("\t")
            target, gold = tokens[1], tokens[2:]
            gold = _expand_gold_truths(target, gold)

            candidates = all_candidates[target]
            best_cand = candidates[0]

            all_segs.append(candidates)
            all_gold.append(gold)

            feats = []
            labels = []
            for ind, seg in enumerate(candidates):
                lv = self._get_label(gold, seg)
                fv = self._get_features_for_segmentation(seg, best_cand)

                feats.append(fv)
                labels.append(lv)

            all_features.append(feats)
            all_labels.append(labels)

        print("Number of hashtags / Data size: ", len(all_features))
        print("Done extracting features")
        return all_features, all_labels, all_segs, all_gold

    def _get_label(self, gold, seg):
        ed = -1*min([editdistance.eval(g.lower(), seg.lower()) for g in gold])

        if self.model == "mse_multi" or self.model == "mr_multi":
            single_word = (max([len(g.split()) for g in gold]) <= 1) * 1.0
            return [ed, 1 - single_word]

        elif self.model == "mse" or self.model == "mr":
            return ed

    def _get_features_for_segmentation(self, seg, best_cand):
        multi_features = self._get_multi_features(seg)
        single_features = self._get_single_features(seg)

        if self.model == "mse_multi" or self.model == "mr_multi":
            padded_features = [0.0] * (len(multi_features) - len(single_features))
            single_features.extend(padded_features)

            target_features = self.hashtag.get_features(seg, best_cand)
            return multi_features, single_features, target_features
        
        elif self.model == "mse" or self.model == "mr":
            return multi_features + single_features

    def _get_single_features(self, seg):
        return self.lm_kn.get_features(seg.lower())

    def _get_multi_features(self, seg):
        seg_org = seg
        tokens_org = seg_org.split()

        seg = seg.lower()
        tokens = seg.split()

        # LM and count features
        multi_features = []
        multi_features.extend(self.lm_gt.get_features(seg))
        multi_features.append(self.google_counts.get_features(seg))
        multi_features.append(self.twitter_counts.get_features(seg))

        # Length featueres
        len_features = [0] * 20
        for token in tokens:
            if len(token) <= 20:
                len_features[len(token) - 1] += 1.0
            else:
                len_features[19] += 1.0
        multi_features.extend(len_features)

        # Linguistic and word-shape features
        multi_features.extend(self.ws.get_features(seg_org, tokens_org))
        multi_features.extend(self.ner.get_features(seg, tokens))
        multi_features.append(self.urban_dict.get_features(seg, tokens))
        multi_features.append(sum([(self.dict.check(token) or token in self.urban_dict.urban_words) * len(token)
                                   for token in tokens]) * 1.0 / len(tokens))
        return multi_features


def _expand_gold_truths(target, gold_truths):
    gold_truths_final = []
    for gd in gold_truths:
        gd = " ".join([g.strip() for g in gd.split()])
        if "".join(gd.lower().split()) == target.lower():
            gold_truths_final.append(gd)
        else:
            if "'" in gd:
                gd_wo_punc = gd.replace("'", " ")
                gd_w_punc = gd.replace("'", "")
                gold_truths_final.extend([gd_w_punc, gd_wo_punc])
            elif "-" in gd:
                gd_wo_punc = gd.replace("-", " ")
                gd_w_punc = gd.replace("-", "")
                gold_truths_final.extend([gd_w_punc, gd_wo_punc])
            elif "." in gd:
                gd_wo_punc = gd.replace(".", "")
                gold_truths_final.extend([gd_wo_punc])
    return gold_truths_final
