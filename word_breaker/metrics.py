def mean_reciprocal_rank(gold, top_segs):
    """Calculates mean reciprocal rank metric.

    Parameters
    ----------
    gold : List
        Gold-truth segmentations
    top_segs : List
        Top segmentations.
    """

    rr = 0
    for gt, segs in zip(gold, top_segs):

        gt = set([g.lower().strip() for g in gt])

        is_seg = [seg.lower() in gt for seg in segs]
        rank = (is_seg.index(True) + 1) if any(is_seg) else 0
        rr += rank if rank == 0.0 else 1.0 / rank

    return rr / len(top_segs)


def accuracy(k, gold, top_segs):
    """Calculates accuracy at k metric.

    Parameters
    ----------
    k : int
        Accuracy of top-k candidates
    gold : List
        Gold-truth segmentations
    top_segs : List
        Top segmentations.
    """

    correct = 0
    for gt, segs in zip(gold, top_segs):

        gt = set([g.lower().strip() for g in gt])

        segs = set([s.lower() for s in segs[:k]])
        if len(segs.intersection(gt)) > 0:
            correct += 1

    return correct*1.0 / len(top_segs)


def fscore(k, gold, top_segs):
    """Calculates F-score at k metric.

    Parameters
    ----------
    k : int
        Accuracy of top-k candidates
    gold : List
        Gold-truth segmentations
    top_segs : List
        Top segmentations.
    """

    fscore = 0
    for gt, segs in zip(gold, top_segs):

        gt = [set(g.lower().strip().split()) for g in gt]
        segs = [set(s.lower().split()) for s in segs[:k]]

        fscore_golds = []
        for seg in segs:
            for g in gt:

                correct = seg.intersection(g)
                prec = (1.0*len(correct)) / len(seg)
                recall = (1.0*len(correct)) / len(g)

                prec_recall_sum = prec + recall
                fscore_one = 0 if prec_recall_sum == 0 else (2*prec*recall) / prec_recall_sum
                fscore_golds.append(fscore_one)

        fscore += max(fscore_golds)

    return fscore*1.0 / len(top_segs)
