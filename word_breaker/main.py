import kenlm
import argparse
from word_breaker import WordBreaker, SegNode
from metrics import accuracy, fscore, mean_reciprocal_rank


def segment_word(target, topk, lm):
    """Extracts top-k segmentations of a hashtag using beam search algorithm.

    Parameters
    ----------
    target : str
        Input hashtag to be segmented.
    topk : int
        Value of k in top-k.
    lm : Object
        Language model
     """

    node = SegNode()
    beam_search = WordBreaker(target, topk, lm)
    beam_search.search(node)
    return beam_search.get_topk()


def expand_gold_truths(target, gold_truths):
    """Deals with gold-truth segmentations with punctuation.

    For example: Human segmentation for a hashtag like "#oneslife" can be "one's life".
    However, during segmentation, we are not adding any new characters. So, we consider
    "one s life" and "ones life" as gold-truth.

    Parameters
    ----------
    target : str
        Input hashtag to be segmented.
     gold-truths : List
         Gold-truth segmentations.
     """

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


def main(args):
    """Runs word-breaker to extract top-k segmentations

    Parameters
    ----------
    args : Object
        Command line arguments.
    """

    k = args.topk

    print("Loading language model.")
    language_model = kenlm.LanguageModel(args.lm)
    print("Done.")

    print("Extracting top segmentations.")
    all_gold_truths = []
    all_top_segmentations = []
    for line in open(args.input):

        tokens = [token.strip() for token in line.strip().split('\t')]
        target, gold_truths = tokens[1], tokens[2:]
        gold_truths = expand_gold_truths(target, gold_truths)

        candidates = segment_word(target, k, language_model)
        all_top_segmentations.append(candidates)
        all_gold_truths.append(gold_truths)

    if args.output is not None:
        fp = open(args.output, 'w')
        for segs in all_top_segmentations:
            target = "".join(segs[0].split())
            fp.write(target + "\t" + "\t".join([seg.strip() for seg in segs]) + "\n")
        fp.close()

    print("MRR:", mean_reciprocal_rank(all_gold_truths, all_top_segmentations))
    print("Accuracy@1:", accuracy(1, all_gold_truths, all_top_segmentations))
    print("Accuracy@2:", accuracy(2, all_gold_truths, all_top_segmentations))
    print("Fscore@1:", fscore(1, all_gold_truths, all_top_segmentations))
    print("Fscore@2:", fscore(2, all_gold_truths, all_top_segmentations))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs word-breaker to extract top-k segmentations of a hashtag.')
    parser.add_argument('--k', help='Value of k in top-k.', type=int, dest='topk')
    parser.add_argument('--lm', help='Path to language model.', type=str, dest='lm')
    parser.add_argument('--out', help='Path to top-k candidates file. \n'
                                      'The output file is tab seperated. The format is: \n'
                                      '<hashtag without #> <tab separated top-k candidates>.',
                        dest='output', type=str)
    parser.add_argument('--input', help='Path to input hashtags file.\n'
                                        'The input file is tab seperated. The format is: \n'
                                        '<tweet> <hashtag without #> <tab separated gold-truth segmentations>.',
                        dest='input', type=str)
    args = parser.parse_args()
    main(args)
