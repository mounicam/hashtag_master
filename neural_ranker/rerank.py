def rerank(segs, segs_feats, model, type):
    
    score_map = {}
    for seg in segs:
        score_map[seg] = 0.0

    if type.startswith("mr"):

        test_x = segs_feats

        if type == "mr_multi":

            multi_x, single_x, hashtag_x = [], [], []
            for sf1 in segs_feats:
                multi_x.append(sf1[0])
                single_x.append(sf1[1])
                hashtag_x.append(sf1[2])

            test_x = (multi_x, single_x, hashtag_x)

        predicted_scores = model.predict(test_x)

        for ind, s1 in enumerate(segs):
            score = predicted_scores[ind]
            score_map[s1] += score

        return sorted(score_map.keys(), key=score_map.__getitem__, reverse=True)

    else:

        test_x = []

        if type == "mse":
            for i, sf1 in enumerate(segs_feats):
                for j, sf2 in enumerate(segs_feats):
                    if i != j:
                        test_x.append(sf1 + sf2)

        else:
            multi_x, single_x, hashtag_x = [], [], []
            for i, sf1 in enumerate(segs_feats):
                for j, sf2 in enumerate(segs_feats):
                    if i != j:
                        multi_x.append(sf1[0] + sf2[0])
                        single_x.append(sf1[1] + sf2[1])
                        hashtag_x.append(sf1[2])

            test_x = (multi_x, single_x, hashtag_x)

        predicted_scores = model.predict(test_x)

        return _greedy(segs, predicted_scores)


def _greedy(segs, predicted_scores):
        scores = {}
        score_map = {}
        for seg in segs:
            score_map[seg] = 0.0
            scores[seg] = 0.0

        count = 0
        for i, s1 in enumerate(segs):
            for j, s2 in enumerate(segs):
                if s1 != s2:
                    score = predicted_scores[count]
                    scores[s1] += score
                    score_map[(s1, s2)] = score
                    count += 1

        segs = set(segs)
        top_segs = []

        while len(segs) > 0:
            sorted_segs = sorted(scores.keys(), key=scores.__getitem__, reverse=True)
            best = sorted_segs[0]

            top_segs.append(best)
            for s in segs:
                if s != best:
                    scores[s] = scores[s] - score_map[(s, best)]

            del scores[best]
            segs.remove(best)

        return top_segs
