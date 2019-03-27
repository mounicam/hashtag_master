import sys
import math
import nltk
import heapq
import kenlm

sys.setrecursionlimit(100000)


class WordBreaker:
    def __init__(self, target, k, lm, order=3):
        """Beam search algorithm to extract the top-k candidate segmentations for a hashtag.

        The implementation is based on "Microsoft Word-Breaker".
        https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/URLwordbreaking.pdf

        Parameters
        ----------
        target : str
            Input hashtag to be segmented.
        k : int
            Number to top candidates.
        lm : KenLM language model object
            Language model used to compute cost of each segmentation during beam search.
        order: int, optional
            Order of language model.
        """
        self.k = k
        self.lm = lm
        self.order = order
        self.target = target

        self.heap = []
        self.topk_segs = []
        self.beam_bucket_by_len = [0] * (len(target) + 1)

    def _score_lm(self, tokens):
        return [prob for prob, _, _ in self.lm.full_scores(' '.join(tokens), eos=False, bos=False)][-1]

    def _get_score(self, context, curr_word):
        phrase = (context + ' ' + curr_word).strip().lower()
        tokens = phrase.split()
        if len(tokens) < self.order:
            return self._score_lm(tokens)
        else:
            return self._score_lm(list(nltk.ngrams(tokens, self.order))[-1])

    def search(self, node):
        """Run beam search.

        Parameters
        ----------
        node : SegNode
            Object storing partial segments.
        """

        for i in range(node.pointer + 1, len(self.target) + 1):
            curr_word = self.target[node.pointer:i]

            score = self._get_score(node.seg, curr_word)
            seg = (node.seg + ' ' + curr_word).strip()

            child = SegNode(score + node.score, i, seg)
            heapq.heappush(self.heap, child)

        while self.beam_bucket_by_len[-1] < self.k and self.heap:
            new_node = heapq.heappop(self.heap)

            if new_node.pointer == len(self.target):
                self.topk_segs.append([new_node.seg, new_node.score])
                self.beam_bucket_by_len[new_node.pointer] += 1

            elif self.beam_bucket_by_len[new_node.pointer] < self.k:
                self.beam_bucket_by_len[new_node.pointer] += 1
                self.search(new_node)

            else:
                continue

    def get_topk(self):
        """Get top k segmentations using from beam search

        Returns
        ----------
        List
            Top k segmentations in a sorted order. The first one is the best segmentation
            according to beam search.
        """

        return [s[0] for s in self.topk_segs]


class SegNode:
    """Stores partial segment information.

    Parameters
    ----------
    score : float
        Partial segmentation score.
    pointer : int
        Position of current word.
    seg : str
        Partial segments of a hashtag.
    """

    def __init__(self, score=0, pointer=0, seg=''):
        self.seg = seg
        self.score = score
        self.pointer = pointer

    def __eq__(self, other):
        return self.score == other.score

    def __ne__(self, other):
        return not (self==other)

    def __lt__(self, other):
        return self.score > other.score

    def __gt__(self, other):
        return self.score < other.score

    def __le__(self, other):
        return self.score <= other.score

    def __ge__(self, other):
        return self.score >= other.score
