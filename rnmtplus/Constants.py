PAD = 0
BOS = 1
EOS = 2
UNK = 3

PAD_WORD = '<BLANK>'
UNK_WORD = '<UNK>'
BOS_WORD = '<BOS>'
EOS_WORD = '<EOS>'

# The index of a missing attribute value (e.g. "NA" in the Kaggle data, preprocess.py -missing_values): the
# losses ignore it (ignore_index) and the evaluation leaves it out.
MISSING = -1
