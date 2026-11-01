# Multimodal Sequential Fashion Attribute Prediction

Code for

> Hasan Sait Arslan, Kairit Sirts, Mark Fishel, Gholamreza Anbarjafari.
> **Multimodal Sequential Fashion Attribute Prediction.** *Information* 2019, 10(10), 308.
> https://doi.org/10.3390/info10100308

The repository contains the code of every experiment in the paper: the proposed Sequential model and the
baselines (MultiClass, Binary MultiLabel, MultiTask, Attentive MultiTask, Self-Attentive MultiTask) on the
Rakuten, Kaggle and Amazon datasets, with text-only, image-only and text+image input. One code base
implements all models; the settings of each experiment are in a file in `experiments/`.

## Repository layout

```
.
├── run.py            runs a stage of an experiment with the settings in its file
├── build_vocab.py    builds the vocabularies
├── preprocess.py     converts the data files into data.pt
├── train.py          trains a model
├── translate.py      predicts attribute chains with a trained model
├── evaluate.py       computes the error rates of the predictions
├── DataLoader.py     batches of images, titles and attribute chains
├── kaggle_data.py    splits the Kaggle data with a seed and prepares it (train.py -kaggle_dir)
├── rnmtplus/         the models (Models.py), prediction (Translator.py), RNMT+ layers and attention
├── experiments/      the settings of the 42 experiments: <dataset>/<input>/<model>.json;
│                     <dataset>_fixed/: the same with the fixes for a fair comparison;
│                     kaggle_seeded/: the Kaggle experiments on a split drawn with a seed, with these fixes
├── analysis/         the analysis scripts of the paper
└── tests/            smoke_test.py: an end-to-end run of every experiment on synthetic data;
                      reproducibility_test.py: the same seed gives the same results
```

## Models

| Paper (section) | `model_type` | Output layer |
|---|---|---|
| MultiClass (4.3.1) | `multiclass` | one softmax over all attribute chains seen in training |
| Binary MultiLabel (4.3.2) | `multilabel` | one sigmoid per attribute value (BCE loss); highest-scoring value per attribute at test time |
| MultiTask (4.3.2) | `multitask` | one softmax classifier per attribute |
| Attentive MultiTask (4.3.2) | `multitask_att` | MultiTask + per-attribute multi-head attention over the image and text features (Eq. 7) |
| Self-Attentive MultiTask (4.3.2) | `multitask_slfatt` | Attentive MultiTask + the attribute representations added to the attended keys and values (Eq. 8) |
| Sequential (3) | `sequential` | RNMT+ decoder that generates the chain one attribute at a time |

All models use the same encoders (Section 3.1): ResNet-50 features for the image (`img`) and the RNMT+ GRU
encoder for the title (`txt`). The baselines average-pool and concatenate the two encodings (Section 4.3).

## Experiments and paper tables

| Dataset | Models | Input |
|---|---|---|
| Rakuten | all six | `txt`, `img`, `txt+img` |
| Kaggle | all six | `txt`, `img`, `txt+img` |
| Amazon | `multiclass`, `sequential` (the category structure is unknown, Section 5) | `txt`, `img`, `txt+img` |

- `txt+img`: Table 2 (full-chain accuracy) and Tables 3–4 (per-attribute accuracy).
- `txt` and `img`: Tables 6 and 7. The paper reports MultiClass, Attentive MultiTask and Sequential for these
  settings; the settings files exist for all six models.
- Table 5 (seen vs. unseen attribute chains on Kaggle): `analysis/seen_unseen.py`.

A settings file, for example `experiments/kaggle/txt+img/multitask_att.json`, holds the data files and the
options of the scripts that were used for the experiment, the folder of the original experiment code
(`experiment`), and the checkpoint the paper's test results were computed with (`paper_checkpoint`).

## Running an experiment

```bash
S=experiments/kaggle/txt+img/multitask_att.json
D=/data/kaggle          # the data files: train.img, train.enc.tit, train.lbl, ... (see Data)
I=/data/kaggle/images
W=work/kaggle/txt+img/multitask_att
python run.py $S vocab      -data_dir $D -work_dir $W   # vocabularies -> $W/*.json
python run.py $S preprocess -data_dir $D -work_dir $W   # -> $W/data.pt
python run.py $S train      -data_dir $D -work_dir $W -image_dir $I -seed 1
python run.py $S translate  -data_dir $D -work_dir $W -image_dir $I -split valid -checkpoint $W/models/model.9.chkpt
python run.py $S evaluate   -data_dir $D -work_dir $W -split valid
```

`run.py` builds the command line of the stage script from the settings file and prints it; `-dry_run` only
prints it, so the scripts can also be run directly. Options after the stage are passed on to the script and
override the settings file. `-split test` (the default) predicts and evaluates the test set
(`$W/pred_test.txt`); `-split valid` the validation set (`$W/pred.txt`).

Every run is seeded: `train.py` needs `-seed`, and `translate.py` uses the seed the checkpoint was trained with
(or `-seed`). The seed sets Python's string hashes (`PYTHONHASHSEED`: the script starts again with it),
Python's `random`, NumPy and PyTorch on the CPU and the GPUs (the initial weights, dropout, the order of the
training data, the random image crops and flips) and the processes that load the images. In `translate.py`, the
random crops and flips of `-test_crop` are drawn per product, so the predictions do not depend on the batches
or the number of processes. In training, as in the original code, they come from the generators seeded with
`-seed`: a run repeats exactly with the same `-num_workers` and with `-deterministic` (otherwise cuDNN picks
the fastest algorithms, which can give slightly different numbers).

As in the paper's runs, `train` trains for up to 500 epochs and saves a checkpoint after every epoch
(`$W/models/model.<epoch>.chkpt`, `-save_mode all`); the logs are `$W/log.train.log` and
`$W/log.valid.log`. In the paper, training on Rakuten and Kaggle was stopped when the validation accuracy had
not improved for three epochs, and the Amazon models were trained for a fixed number of epochs (Section 4.4).
To stop automatically and keep only the best checkpoint, `$W/models/model.best.chkpt`:

```bash
python run.py $S train -data_dir $D -work_dir $W -image_dir $I -seed 1 -save_mode best -patience 3
```

`analysis/best_epoch.py $W/log.valid.log -metric accuracy` finds the best epoch in a log (for `multilabel`,
which logs no accuracy, the lowest loss).

The checkpoints of the original experiments load as they are: `translate.py` reads the model type from
`-model_type` (which `run.py` passes) when the checkpoint does not record it. They record no seed: give one
with `-seed`. Predict with the `data.pt`
that a checkpoint was trained with, which holds its vocabularies; options after the stage override those of
`run.py`, for example `-vocab <experiment>/dataset/data.pt`.

## The Kaggle data split with a seed

The Kaggle data has no predefined validation and test sets: the paper drew one split at random, which is in the
data files of its runs. The settings in `experiments/kaggle_seeded/` run the 18 Kaggle experiments on a split
drawn with a seed instead. The seed sets the split, the tokenizer, the order of the training data, the image
crops and flips, the initial weights and dropout, so a run can be repeated exactly.

```bash
S=experiments/kaggle_seeded/txt+img/sequential.json
K=/data/kaggle_fashion   # the Kaggle data: styles.csv and images/
W=work/kaggle_seeded/txt+img/sequential/seed1
python run.py $S train     -data_dir $K -work_dir $W -seed 1
python run.py $S translate -data_dir $K -work_dir $W    # the test set, with $W/models/model.best.chkpt
python run.py $S evaluate  -data_dir $K -work_dir $W
```

`train` (`train.py -kaggle_dir $K -seed 1`) first prepares the data in `$W/data` (`kaggle_data.py`):

- the 44,446 products of `styles.csv`, with the titles and attribute chains of the paper's data files (spaces in
  a value become `_`, an empty value `None`), without product 28319, a test record ("RTV_LoadTest") whose
  attributes do not match its image (`-exclude_ids`);
- the split: the products shuffled with `numpy.random.RandomState(seed)`, whose numbers are the same in every
  NumPy version. The first 5,000 are the test set, the next 5,000 the validation set, the other 34,445 the
  training set (`-valid_size` and `-test_size` give other sizes, or fractions if below 1);
- a SentencePiece model trained on the training titles with the settings of the paper's model (unigram, 4,000
  pieces, all characters), and the titles encoded with it;
- the vocabularies of the training set and `data.pt`; `split.json` records the settings.

Then it trains with the same seed:

- Python's string hashes (`PYTHONHASHSEED`), `random`, NumPy and PyTorch (CPU and GPUs: the initial weights
  and dropout) are seeded with it;
- the training data of epoch e is shuffled with `numpy.random.RandomState([seed, e])`;
- the random crop and flip of a training image are drawn from the seed, the epoch and the product, and the
  processes that load the images are seeded from the seed, the epoch and their number, so the batches do not
  depend on the number of processes (`-num_workers`);
- deterministic algorithms are used: cuDNN's, and in PyTorch 1.8 and later those of all operations.

`translate` uses the seed of the checkpoint.

The settings use the fixes of the next section. The validation and test images are resized (shorter side 256)
and center-cropped to 224 × 224, without randomness; the training images keep the random resized crop and flip
of the paper's runs. Training stops when the full-chain accuracy of the predicted validation chains has not
improved for three epochs and keeps the best checkpoint. Runs with the same seed
on the same type of GPU and the same software versions give the same files, weights, logs and predictions;
another GPU type or other versions of PyTorch, Pillow or SentencePiece can give other numbers. The five
products of the Kaggle data without an image get a blank image, as in the paper's runs.

## Settings for a fair comparison

The settings in `experiments/<dataset>/` reproduce the paper's runs, with training details that differ between
the models or from the description of the paper (Section 4.4). The settings in `experiments/<dataset>_fixed/`
(the paper's splits of Rakuten, Kaggle and Amazon) and in `experiments/kaggle_seeded/` train all the models of a
dataset in the same way, as the paper describes, without changing the models:

| Training detail | The paper's runs | The fixed settings | Option |
|---|---|---|---|
| Training crop | `RandomCrop` for some Rakuten and Amazon models, `RandomResizedCrop` for the others | the random resized crop and flip that the paper describes, for all | `-train_crop random_resized_crop` |
| Validation and test images | random crop and flip | resized and center-cropped, without randomness | `-valid_crop` and `-test_crop center_crop` |
| Model selection and early stopping | the accuracy of the training targets of each model (tokens given the gold previous ones, attributes or chains), the loss for Binary MultiLabel | the full-chain accuracy of the predicted validation chains, for all models | `-select_by chain_accuracy` |
| Pretrained ResNet | its BatchNorm statistics are updated in training | entirely frozen, as the paper describes | `-frozen_resnet` |
| Title padding in the baselines | part of the average pooling and of the attention | left out, as in the attention of the Sequential model | `-mask_padding` |
| Regularization of Vaswani et al. | dropout 0.1 | dropout 0.1 and label smoothing 0.1 | `-label_smoothing 0.1` |
| Amazon epochs | the checkpoint of the 8th epoch (`model.7.chkpt`) | 7 epochs and the last model, as the paper describes | `-epoch 7 -save_mode last` |
| Token vocabularies | the most frequent token shares the index of `<UNK>` | every token has its own index | `build_vocab.py -distinct_unk` |
| Missing attribute values (Kaggle: `NA` in `styles.csv`, `None` for an empty field; 0.8% of the products) | values that the models learn to predict and that are scored | left out of the vocabularies, the losses (index -1, `ignore_index`) and the scores; after a missing value, the Sequential decoder gets `<UNK>` as input | `"missing_values": ["NA", "None"]` (`-missing_values NA,None`) |

The Kaggle settings also leave out product 28319, a test record ("RTV_LoadTest") whose attributes do not match
its image (`"exclude_ids"`): for `experiments/kaggle_fixed/`, `run.py` writes the data files without it to
`WORK/data`, which all stages then read.

The architecture, the optimizer, the learning rate schedule, the batch size and the early stopping after three
epochs without improvement are those of the paper in all settings, and every dataset uses the vocabularies of
its own training data. The paper reports one run per model; to compare the models, run each with several seeds
and report the mean and standard deviation of the error rates:

```bash
S=experiments/rakuten_fixed/txt+img/sequential.json
for seed in 1 2 3; do
  W=work/rakuten_fixed/txt+img/sequential/seed$seed
  python run.py $S vocab      -data_dir $D -work_dir $W
  python run.py $S preprocess -data_dir $D -work_dir $W
  python run.py $S train      -data_dir $D -work_dir $W -image_dir $I -seed $seed
  python run.py $S translate  -data_dir $D -work_dir $W -image_dir $I   # with $W/models/model.best.chkpt
done
python analysis/seed_summary.py -n_attributes 7 -gold $D/test.lbl work/rakuten_fixed/txt+img/sequential/seed*
```

With `experiments/kaggle_seeded/`, every seed also draws its own split: `analysis/seed_summary.py -n_attributes 8
WORK ...` reads the test set of each run from `WORK/data/test.lbl`.

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

| Purpose | Script |
|---|---|
| Table 5 and Section 5.1: error rates on test items whose chain was seen in training and on the others | `analysis/seen_unseen.py` |
| Section 5.2: test items predicted correctly from text only but not from text+image | `analysis/find_errors.py` |
| Test items where MultiClass, MultiTask and Sequential predict three different values of an attribute (Kaggle: base colour) | `analysis/compare_predictions.py` |
| Heat maps of the image attention of the Sequential model (`translate.py -save_attention`) | `analysis/visualize_heatmap.py` |
| Chain frequencies, attribute transitions, chain lengths | `analysis/label_stats.py` |
| The best epoch in a training log | `analysis/best_epoch.py` |
| The mean and standard deviation of the error rates of runs with different seeds | `analysis/seed_summary.py` |

## Differences from the code of the paper's runs

The experiments of the paper were run with a copy of the code for each experiment; the copies are in commit
`8f21721` (folder `experiments/`). This version merges them. With the settings file of an experiment, it
computes what the copy of that experiment computed (see [Verification](#verification)). The differences:

**Faster, and less GPU memory**

- No backpropagation through the pretrained ResNet-50, whose weights are not trained (the original code
  computed their gradients and discarded them).
- No gradient bookkeeping in validation and prediction.
- The images are read and transformed in worker processes (`-num_workers`, default 4) while the GPU computes.
- Prediction in batches: the models that predicted one product at a time batch titles of the same length,
  which need no padding. MultiClass and Binary MultiLabel keep the batches of the original code
  (`-batch_size 32`), because the padding of a batch enters their average pooling.
- The attention weights are no longer written to disk at every decoding step; `translate.py -save_attention
  DIR` saves those of the last step, for `analysis/visualize_heatmap.py`.
- `-save_mode best` keeps only the best checkpoint (the original code ignored this option and saved every
  epoch), and `-patience N` stops training after N epochs without improvement.
- cuDNN picks the fastest convolution algorithms (`-deterministic` for reproducible ones); prediction does not
  download the ImageNet weights, which are in the checkpoint.

On a V100 with PyTorch 1.1, a training epoch of 3,200 products took 4 times less time for the Kaggle image
models (Attentive MultiTask with text and image: 152 s → 38 s; most of the gain is decoding the large Kaggle
images in parallel) and 1.4 times less for the Rakuten text+image Sequential model; text-only training is as
fast as before. Predicting a test set of 5,000 products took 6 to 16 times less time (Rakuten text+image
Sequential: 493 s → 60 s).

**PyTorch 1.1 to 2.x**

- `pack_padded_sequence` gets the lengths on the CPU (required since PyTorch 1.7).
- `data.pt` and the checkpoints, which contain the options as an `argparse.Namespace`, are loaded with
  `weights_only=False` (needed since PyTorch 2.6).
- ResNet-50 gets the ImageNet weights of the experiments (`resnet50-19c8e357.pth`) in every torchvision version.
- Deprecated calls were replaced (`Variable(volatile=...)`, `size_average=False`).
- The computations stay in float32 on Ampere and newer GPUs, where PyTorch would use the less precise
  TensorFloat-32 in cuDNN; `-tf32` allows it, which is faster.
- `build_vocab.py` sorts by frequency as NumPy 1.x did; NumPy 2 orders equal frequencies differently, which
  would give words other indices.

**Kept from the original code, although they affect the results** (the settings in `experiments/<dataset>_fixed/`
and `experiments/kaggle_seeded/` change them: see [Settings for a fair comparison](#settings-for-a-fair-comparison))

- The random crop of the images (`RandomCrop` or `RandomResizedCrop`, as in each experiment) and the random
  flip are applied in validation and prediction too, so the predictions of the image models depend on the
  seed.
- The BatchNorm layers of the ResNet are in training mode during training.
- The baselines average the title encodings over the padding positions of a batch.
- The first update uses Adam's default learning rate; the schedule is applied after each update.
- `build_vocab.py -mode tokens` gives the most frequent token index 3, the index of `<UNK>`.

**Other changes**

- Every run is seeded (see [Running an experiment](#running-an-experiment)); the original code drew its
  random numbers without a seed.
- The settings of the Kaggle text+image baselines build the title vocabulary from the SentencePiece-encoded
  titles that the models read (`train.enc.tit.json`), as the other Kaggle settings do; the scripts of the
  paper's runs used another vocabulary file, `train.tit.json`.
- The Rakuten text+image Sequential model trains; its copy failed with "too many values to unpack".
- `evaluate.py` divides by the number of products instead of 5,000, and stops if a prediction file is shorter
  than the gold file.
- `train.py` creates the checkpoint directory; `-batch_size` and `-d_model` default to 32 and 1024, the values
  of all experiments; options that had no effect were removed.
- The logs of Binary MultiLabel have the header `epoch,loss`; the text and text+image versions wrote
  `epoch,loss,ppl,accuracy` above two columns.
- `analysis/compare_predictions.py` leaves out a condition of `extract_diff.py` that depended on the iteration
  order of a Python 2 set; on Kaggle, it lists 150 products instead of 35.

## Verification

- **Original code and this code, PyTorch 1.1 on a V100, all 42 experiments**, on synthetic data with
  deterministic image transforms and cuDNN: the vocabularies, `data.pt`, the weights and logs after one seeded
  epoch of training, the predictions (also with the checkpoint of the original code) and the error rates are
  identical.
- **The 42 checkpoints of the paper, real data**: the predictions of this code and of the original code are the
  same on the first 500 test products of every experiment. For the 14 text-only experiments, the predictions on
  the whole test sets are those saved from the paper's runs.
- `evaluate.py` and the analysis scripts give the outputs of the original scripts on the predictions saved from
  the paper's runs (Table 5 included), and `build_vocab.py` the vocabularies of the original scripts (the
  Rakuten files from 2019 also have trailing spaces, written by Python 2's `json`).
- **Python 3.12, PyTorch 2.14.1, torchvision 0.29.1 on an A100**: `tests/smoke_test.py` passes for all 42
  settings files, and the text-only models give the saved predictions of the paper on the whole test sets. The
  image models agree with the original stack on 93–100% of the first 500 test products: newer Pillow versions
  decode the JPEG images slightly differently (Pillow 13.0.0 and 6.2.2 differ on all of 60 Rakuten test images
  and on 26 of 60 Kaggle images).
- **The Kaggle data split with a seed**: `kaggle_data.py` reads the titles and attribute chains of the paper's
  data files for all 44,446 products of `styles.csv`. On both stacks above, `tests/reproducibility_test.py`
  passes for the 18 settings, and two runs of `experiments/kaggle_seeded/txt+img/sequential.json` on the real
  data with seed 1 (one epoch, with 4 and with 7 processes loading the images) give identical data, weights,
  logs and test predictions. A seed gives the same split with NumPy 1.19 and 2.5; SentencePiece 0.2.0 and 0.2.2
  learn different models from the same titles.
- **Seeds**: on both stacks, two runs of six of the paper's settings with the same seed (and `-deterministic`)
  give identical weights, logs and predictions, the predictions also with another number of processes loading
  the images; with the same seed, the training is that of the code before every run was seeded (10 settings).
  The scripts run with `PYTHONHASHSEED` set to the seed.
- **Settings for a fair comparison**: on both stacks, `tests/smoke_test.py` passes for all 102 settings files and
  `tests/reproducibility_test.py` for the 18 seeded Kaggle settings with the fixes. With `-mask_padding`, a
  title gets the same outputs with and without padding in its batch. With `-select_by chain_accuracy`, the full-
  chain accuracy logged in training is that of `evaluate.py` for the validation predictions of `translate.py`
  with the kept checkpoint (also on the real Kaggle data). With `-frozen_resnet`, the BatchNorm layers keep the
  ImageNet statistics. With the default options, the results are those of the code before these options (10
  settings). With `missing_values`, the missing values of the Kaggle data are left out of the vocabularies, the
  losses (checked against the losses computed without them) and the scores; the 274 missing values of the
  training set of seed 1 became 274 ignored targets, and the Sequential decoder gets the same input for a
  missing value as for `<UNK>`.

## Requirements

Python 3.6 or later, PyTorch 1.1 or later with torchvision, NumPy, Pillow and tqdm; a GPU (`-no_cuda` runs on
the CPU, slowly). Tested with Python 3.7, PyTorch 1.1.0, torchvision 0.3.0 and Pillow 6.2.2 (versions of
2019, when the experiments were run with Python 3.6 and PyTorch 1.x), and with Python 3.12, PyTorch 2.14.1,
torchvision 0.29.1 and Pillow 13.0.0, which decodes some JPEG images slightly differently (see
[Verification](#verification)). The Kaggle data split with a seed needs SentencePiece (`pip install
sentencepiece`), the heat-map script OpenCV and scikit-image, `analysis/label_stats.py frequency` matplotlib.
`python tests/smoke_test.py` runs every experiment on synthetic data, to check an installation, and
`python tests/reproducibility_test.py` checks that the same seed gives the same results there.

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
