import kenlm


class LMFeatures:
    def __init__(self, lm_paths):
        """Extracts language model features.

        Parameters
        ----------
        lm_paths : List
            List of language model paths
        """

        self.models = [kenlm.LanguageModel(lm) for lm in lm_paths]

    def get_features(self, seg):
        """Gets language model features.

         Parameters
         ----------
         seg : str
             Candidate segmentation.

         Returns
         -------
         List
             Language model probabilities.
         """

        return [m.score(seg.lower(), bos=False, eos=False) for m in self.models]