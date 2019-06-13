"""Paths to all the resources needed to calculate features.

"""

LM_KN = [
    "data/small_kn.bin"
]

LM_GT = [
    "data/small_gt.bin" 
]

RESOURCES = {
    "lm_gt": LM_GT,
    "lm_kn": LM_KN,
    "wiki": "data/wiki_titles.txt",
    "urban": "data/urban_dict_words_A_Z.txt",
    "twitter": "data/twitter_counts.tsv",
    "google":  "data/google_counts.tsv",
}


def get_resources():
    return RESOURCES
