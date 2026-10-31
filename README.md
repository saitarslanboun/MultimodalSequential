# Multimodal Sequential Fashion Attribute Prediction

Code for

> Hasan Sait Arslan, Kairit Sirts, Mark Fishel, Gholamreza Anbarjafari.
> **Multimodal Sequential Fashion Attribute Prediction.** *Information* 2019, 10(10), 308.
> https://doi.org/10.3390/info10100308

The repository contains the proposed Sequential model and the code of every experiment reported in the
paper: the Sequential model and all baselines (MultiClass, Binary MultiLabel, MultiTask, Attentive MultiTask,
Self-Attentive MultiTask) on the Rakuten, Kaggle and Amazon datasets, with text-only, image-only and
text+image input.

## Repository layout

```
.
├── train.py, translate.py, preprocess.py, DataLoader.py, rnmtplus/
│                      Sequential model as originally released (Rakuten, text+image)
└── experiments/       code of all 42 experiments in the paper
    ├── RakutenDatasetExperiments/{TXT,IMG,TXT+IMG}/<model>/
    ├── KaggleDatasetExperiments/{TXT,IMG,TXT+IMG}/<model>/
    │   └── error_analysis/
    └── AmazonDatasetExperiments/{TXT,IMG,TXT+IMG}/{MultiClass,EncoderDecoder}/
```

`TXT` = text-only, `IMG` = image-only, `TXT+IMG` = multimodal input.

The folders under `experiments/` are the original experiment folders with data, checkpoints, logs and
predictions removed; the code in them is unchanged. The root-level code is
`experiments/RakutenDatasetExperiments/TXT+IMG/EncoderDecoder/model/` without docstrings and an unused
`validate()` method, and with the fix described under [Known issue](#known-issue).

## Models

| Paper (section) | Folder | Output layer |
|---|---|---|
| MultiClass (4.3.1) | `MultiClass` | one softmax over all attribute chains seen in training |
| Binary MultiLabel (4.3.2) | `MultiLabel` | one sigmoid per attribute value (BCE loss); highest-scoring value per attribute at test time |
| MultiTask (4.3.2) | `MultiLabelMultiClass` | one softmax classifier per attribute |
| Attentive MultiTask (4.3.2) | `MultiLabelMultiClass_Att` | MultiTask + per-attribute multi-head attention over the image and text features (Eq. 7) |
| Self-Attentive MultiTask (4.3.2) | `MultiLabelMultiClass_SlfAtt` | Attentive MultiTask + the attribute representations added to the attended keys and values (Eq. 8) |
| Sequential (3) | `EncoderDecoder` | RNMT+ decoder that generates the chain one attribute at a time |

All models use the same encoders (Section 3.1): ResNet-50 features for the image and the RNMT+ GRU encoder
for the title. The baselines average-pool and concatenate the two encodings (Section 4.3).

## Experiments and paper tables

| Dataset | Models | Input |
|---|---|---|
| Rakuten | all six | `TXT`, `IMG`, `TXT+IMG` |
| Kaggle | all six | `TXT`, `IMG`, `TXT+IMG` |
| Amazon | `MultiClass`, `EncoderDecoder` (the category structure is unknown, Section 5) | `TXT`, `IMG`, `TXT+IMG` |

- `TXT+IMG` folders: Table 2 (full-chain accuracy) and Tables 3–4 (per-attribute accuracy).
- `TXT` and `IMG` folders: Tables 6 and 7. The paper reports MultiClass, Attentive MultiTask and Sequential
  for these settings; the folders contain all six models.
- Table 5 (seen vs. unseen attribute chains on Kaggle): see [Analysis scripts](#analysis-scripts).

## Running an experiment

Every experiment folder has the same structure:

```
<model>/
├── dataset/   vocabulary builders and small dataset-statistics helpers; the data files go here
├── model/     model code: train.py, translate.py, DataLoader.py, preprocess.py, rnmtplus/
└── work/      preprocess.sh, train.sh, translate.sh, translate_test.sh, validate.py, test.py
```

1. Put the data files (see [Data](#data)) into `dataset/` and build the vocabularies there:
   `python build_dictionary.py <file>` writes `<file>.json`, a token vocabulary (used for titles and for
   the Sequential model's labels); `python build_att_dictionary.py train.lbl` writes the label vocabulary
   of the baselines (MultiClass: one class per full chain; multilabel models: one vocabulary per attribute).
   `work/preprocess.sh` shows which files each experiment uses.
2. Set `-image_dir` in `work/*.sh` to your image folder. The scripts still contain the paths on the
   authors' cluster.
3. Run everything from `work/`; the scripts use paths relative to it:

```bash
cd experiments/KaggleDatasetExperiments/TXT+IMG/MultiLabelMultiClass_Att/work
bash preprocess.sh       # data files + vocabularies -> ../dataset/data.pt
mkdir -p models          # train.py does not create it
bash train.sh            # writes models/model.<epoch>.chkpt and log.{train,valid}.log every epoch
bash translate.sh        # dev predictions  -> pred.txt
bash translate_test.sh   # test predictions -> pred_test.txt
python validate.py       # dev: error rate per attribute and for the full chain
python test.py           # the same for the test set
```

`validate.py` and `test.py` read the gold labels from `work/` (`valid.lbl` and `test.lbl`; Amazon:
`valid.cat` and `test.cat`) and assume 5,000 items per set.

`train.sh` trains for up to 500 epochs; there is no automatic early stopping. In the paper, training on
Rakuten and Kaggle was stopped when the validation accuracy had not improved for three epochs, and Amazon
models were trained for a fixed number of epochs (Section 4.4). `work/translate_test.sh` names the
checkpoint selected for each experiment. In some `TXT` and `IMG` folders, `work/translate.sh` still points
to the checkpoint of the folder it was copied from; the paper reports only test results for these settings.

## Data

The data is not part of this repository.

- **Rakuten**: a confidential dataset of Rakuten Fits.Me; it cannot be distributed.
- **Kaggle**: [Fashion Product Images Dataset](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset).
- **Amazon**: Amazon product metadata, category *Clothing, Shoes & Jewelry*, from the
  [AWS Open Data registry](https://registry.opendata.aws/).

Each split is a set of parallel text files with one line per item:

- `*.img`: image file name, relative to `-image_dir`;
- `*.tit`, `*.enc.tit`: product title, tokenised with SentencePiece;
- `*.lbl`: the attribute chain, values separated by spaces (multi-word values joined with `_`).

For Amazon, the training and validation sets are expanded to one line per (item, chain) pair (`etrain.*`,
`evalid.*`), and `valid.cat` / `test.cat` list all chains of an item separated by ` ||| `. A prediction
counts as correct if it equals any of them.

## Analysis scripts

| Purpose | Scripts |
|---|---|
| Table 5: split the test predictions of all six models into seen and unseen chains and score them | `KaggleDatasetExperiments/TXT+IMG/EncoderDecoder/work/`: `extract_seen.py`, `extract_unseen.py`, `check_seen_acc_chain.py`, `check_unseen_acc_chain.py` (they read `../../<model>/work/pred_test.txt` of the sibling folders) |
| Unseen chains on Rakuten and Amazon (Section 5.1) | `RakutenDatasetExperiments/TXT+IMG/EncoderDecoder/work/`: `extract_*.py`, `check_*acc_chain.py`; `AmazonDatasetExperiments/TXT+IMG/EncoderDecoder/work/extract_unseen.py` |
| Test items predicted correctly from text only but not from text+image (Section 5.2) | `KaggleDatasetExperiments/error_analysis/find_errors.py` |
| Test items where MultiClass, MultiTask and Sequential disagree on BASE COLOR | `KaggleDatasetExperiments/TXT+IMG/MultiLabelMultiClass*/work/comparison/extract_diff.py` |
| Image-attention heat maps of the Sequential model | `RakutenDatasetExperiments/TXT+IMG/EncoderDecoder/work/`: `translate_vis.sh`, `visualization/visualize_heatmap.py` |
| Attribute-transition statistics and label frequencies | `dataset/prepare_freqs*.py`, `dataset/get_most_frequent.py`, `dataset/get_max_len.py`, `dataset/extract_subs.py` |

## Requirements

The experiments were run in 2019 with Python 3.6 and PyTorch 1.x with CUDA (at least 1.1, for
`pack_padded_sequence(..., enforce_sorted=False)`; parts of the code call `.cuda()` unconditionally),
torchvision, NumPy, Pillow and tqdm. The code has not been updated for current PyTorch versions; the
root-level code has been checked to run (preprocessing, two epochs of training with the settings of
`work/train.sh`, and decoding, on synthetic data) with Python 3.7, PyTorch 1.1.0 and torchvision 0.3.0 on
an NVIDIA V100 GPU. The heat-map script also needs OpenCV, scikit-image and matplotlib.
A few small helper scripts are Python 2: `check_acc_chain.py`, `extract_subs.py`, `get_max_len.py`,
`prepare_freqs*.py`, and `extract_unseen.py` in the Amazon folder.

## Known issue

In `experiments/RakutenDatasetExperiments/TXT+IMG/EncoderDecoder/model/`, `Decoder.forward` also returns
the attention weights (added for the heat maps), but `RNMTPlus.forward` unpacks only two values, so
training fails with `ValueError: too many values to unpack`. This is fixed in the root-level code. To train
from the experiment folder, change that line in its `rnmtplus/Models.py` to

```python
decoder_output, _, _ = self.decoder(tgt_seq, encoder_outputs, V)
```

The other `EncoderDecoder` folders are not affected.

## Citation

```bibtex
@article{arslan2019multimodal,
  title   = {Multimodal Sequential Fashion Attribute Prediction},
  author  = {Arslan, Hasan Sait and Sirts, Kairit and Fishel, Mark and Anbarjafari, Gholamreza},
  journal = {Information},
  volume  = {10},
  number  = {10},
  pages   = {308},
  year    = {2019},
  doi     = {10.3390/info10100308}
}
```
