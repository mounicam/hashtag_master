import math


class CountFeatures:
    def __init__(self, file_name):
        """Extract count features.

        We use two types of counts : Google ngram counts and Twitter ngram counts.

        Parameters
        ----------
        file_name : str
            Count file name.
        """

        counts = {}
        for line in open(file_name):

            tokens = line.strip().split('\t')

            if len(tokens) > 1:
                target, count = tokens[0].lower().strip(), int(tokens[1])

                if target not in counts:
                    counts[target] = 0
                counts[target] += count

        log_counts = {}
        for k, v in counts.items():
            log_counts[k] = math.log10(v + 1)

        self.counts = log_counts

    def get_features(self, seg):
        """Get count features.

        Parameters
        ----------
        seg : str
            Candidate segmentation.

        Returns
        -------
        int
            Ngram count
        """

        return self.counts.get(seg.lower(), 0)
