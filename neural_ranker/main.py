import argparse
from rerank import *
from metrics import *
from config import get_resources
from features.feature_extractor import FeatureExtractor
from models import mse_ranker, mr_ranker, mse_multi_ranker, mr_multi_ranker


def main(args):

    feature_extractor = FeatureExtractor(get_resources(), args.model)
    train_feats, train_labels, _, _ = feature_extractor.get_features(args.train, args.train_topk)
    test_feats, _, test_segs, test_gold_truths = feature_extractor.get_features(args.test, args.test_topk)

    epochs, lr1, lr2 = 100, 0.01, 0.05

    # Initialize model
    model = None
    if args.model == "mse":
        model = mse_ranker.MSERanker(epochs, lr1)
    elif args.model == "mr":
        model = mr_ranker.MRRanker(epochs, lr1)
    elif args.model == "mse_multi":
        model = mse_multi_ranker.MSEMultiRanker(epochs, lr1, lr2)
    elif args.model == "mr_multi":
        model = mr_multi_ranker.MRMultiRanker(epochs, lr1, lr2)

    # Train model
    model.train(train_feats, train_labels)

    # Rerank top-k segmentations
    top_segmentations = []
    for segs_feats, segs, gds in zip(test_feats, test_segs, test_gold_truths):
        if len(segs) == 1:
            top_segmentations.extend(segs)
        else:
            reranked_segs = rerank(segs, segs_feats, model, args.model)
            top_segmentations.append(reranked_segs)

    # Evaluate metrics
    print("MRR:", mean_reciprocal_rank(test_gold_truths, top_segmentations))
    print("Accuracy@1:", accuracy(1, test_gold_truths, top_segmentations))
    print("Accuracy@2:", accuracy(2, test_gold_truths, top_segmentations))
    print("Fscore@1:", fscore(1, test_gold_truths, top_segmentations))
    print("Fscore@2:", fscore(2, test_gold_truths, top_segmentations))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs our pairwise neural ranker model.')
    parser.add_argument('--train', help='Path to train hashtags file.\n'
                                        'The input file is tab seperated. The format is: \n'
                                        '<tweet> <hashtag without #> <tab separated gold-truth segmentations>.',
                        dest='train', type=str)
    parser.add_argument('--train_topk', help='Path to top-k candidates file of traning dataset. \n'
                                             'The output file is tab seperated. The format is: \n'
                                             '<hashtag without #> <tab separated top-k candidates>.',
                        dest='train_topk', type=str)
    parser.add_argument('--test', help='Path to test hashtags file. The format is same as traning dataset. \n',
                        dest='test', type=str)
    parser.add_argument('--test_topk', help='Path to top-k candidates file of traning dataset. \n'
                                            'The format is same as traning dataset.',
                        dest='test_topk', type=str)
    parser.add_argument('--model', type=str, dest='model', help='Type of model. The input should be one'
                                                                'of the strings: mse, mse_multi, mr, mr_multi')
    args = parser.parse_args()
    main(args)
