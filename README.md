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
├── hm_data.py        the same for the H&M data (train.py -hm_dir)
├── fashiongen_export.py  converts the FashionGen release (HDF5) into one row per product and image files
├── fashiongen_data.py    prepares the FashionGen data with its fixed test set (train.py -fashiongen_dir)
├── resize_images.py  resizes the images in advance, for faster training and prediction
├── rnmtplus/         the models (Models.py), prediction (Translator.py), RNMT+ layers and attention
├── experiments/      the settings of the 42 experiments: <dataset>/<input>/<model>.json;
│                     <dataset>_fixed/: the same with the fixes for a fair comparison;
│                     kaggle_seeded/: the Kaggle experiments on a split drawn with a seed, with these fixes;
│                     hm_seeded/: the same experiments on the H&M data;
│                     fashiongen_seeded/: and on the FashionGen data, with all views of a product
├── analysis/         the analysis scripts of the paper
└── tests/            smoke_test.py: an end-to-end run of every experiment on synthetic data;
                      reproducibility_test.py: the same seed gives the same results;
                      masking_test.py: the models leave out the images of products without one, and
                      the views a product does not have; fashiongen_export_test.py: the FashionGen export
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
python resize_images.py -input $K/images -output $K/images_256    # once (optional, see below)
python run.py $S train     -data_dir $K -work_dir $W -seed 1 -image_dir $K/images_256
python run.py $S translate -data_dir $K -work_dir $W -image_dir $K/images_256    # with model.best.chkpt
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

`resize_images.py` resizes the images once, as the crops of these settings resize them first (the shorter
side to 256, keeping the aspect ratio), and saves them without loss (PNG), with a record `resized.json`. With
`-image_dir $K/images_256`, training and prediction read these copies instead of decoding and resizing the
full-size photos (most are 1080 × 1440 or 1800 × 2400): the images are the same, and so are the results,
with the Pillow and torchvision versions that resized them. On a V100 with 6 CPUs, an epoch of the text+image
Sequential model takes 4.0 instead of 9.9 minutes. Without `-image_dir`, they read `$K/images`. The copies
appear in `$K/images_256` only when all of them are written (`resize_images.py` works in
`$K/images_256.partial`), and training and prediction stop if `-image_dir` does not exist, so no run reads an
unfinished directory.

The settings use the fixes of the next section. The images are resized (shorter side 256, keeping the aspect
ratio); validation and test take the center 224 × 224, without randomness, and training a random 224 × 224
crop and a random flip. Training stops when the full-chain accuracy of the predicted validation chains has not
improved for three epochs and keeps the best checkpoint. Runs with the same seed
on the same type of GPU and the same software versions give the same files, weights, logs and predictions;
another GPU type or other versions of PyTorch, Pillow or SentencePiece can give other numbers. The five
products of the Kaggle data without an image get a blank image, as in the paper's runs.

## The H&M data split with a seed

The settings in `experiments/hm_seeded/` run the 18 experiments on the article catalogue of the
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
data (its use is subject to the competition's rules), split with a seed as the Kaggle data is (previous
section): the same steps, options and fixes.

```bash
S=experiments/hm_seeded/txt+img/sequential.json
H=/data/hm   # the H&M data: articles.csv and images/
W=work/hm_seeded/txt+img/sequential/seed1
python resize_images.py -input $H/images -output $H/images_256    # once (optional)
python run.py $S train     -data_dir $H -work_dir $W -seed 1 -image_dir $H/images_256
python run.py $S translate -data_dir $H -work_dir $W -image_dir $H/images_256    # with model.best.chkpt
python run.py $S evaluate  -data_dir $H -work_dir $W
```

`train` (`train.py -hm_dir $H -seed 1`) prepares the data in `$W/data` with `hm_data.py`, from the 105,542
articles of `articles.csv`:

- the text: the name and the description as they are in `articles.csv`, `<prod_name>. <detail_desc>`, or the
  name alone for the 416 articles without a description;
- the attribute chain: the names of 11 attributes from coarse to fine, with spaces in a name replaced by `_`;
- the image: `images/<first three digits of the id>/<id>.jpg`, or `None` for the 442 articles without one;
- the split, the SentencePiece model, the vocabularies and `data.pt` as for the Kaggle data: 5,000 test and
  5,000 validation articles, each article on its own (the colour variants of a product can fall in different
  sets).

| # | Attribute | Column of `articles.csv` | Values |
|---|---|---|---|
| 1 | index group | `index_group_name` | 5 |
| 2 | index | `index_name` | 10 |
| 3 | product group | `product_group_name` | 19 |
| 4 | garment group | `garment_group_name` | 21 |
| 5 | section | `section_name` | 56 |
| 6 | department | `department_name` | 250 |
| 7 | product type | `product_type_name` | 131 |
| 8 | lightness | `perceived_colour_value_name` | 8 |
| 9 | perceived colour | `perceived_colour_master_name` | 20 |
| 10 | colour | `colour_group_name` | 50 |
| 11 | pattern | `graphical_appearance_name` | 30 |

The data gives no order of the attributes (its columns run partly from fine to coarse: the product type before
the product group, the department before the index). The order above was chosen from the data:

1. An attribute that another one determines comes before it: the index group (determined by the index, the
   section and the department), the index, section and garment group (by the department) and the product group
   (by the product type).
2. Of the orders that keep this rule, the chain adds the least information at each step: the sum of the
   entropies of its prefixes over the articles is the lowest of all orders (an exact search over the 2^11 sets
   of attributes). The same order results on two random halves of the products, with the codes instead of the
   names, and without the articles that have a placeholder.

A few names stand for several H&M codes (22 department names, the section `Ladies Other`, the product type
`Umbrella`); the attributes before them in the chain tell them apart.

`Unknown`, `Undefined` and `undefined` are H&M's placeholders for an unknown value: as `NA` and `None` for the
Kaggle data, they are missing values (`missing_values` in the settings), which are not targets and not scored.
4.5% of the articles have at least one, mostly an unknown garment group. The colour `Other` is a colour of its
own.

The 442 articles without an image stay in the data with their text and targets, but the image models leave out
their image (`-mask_missing_images` in the settings): its features are zero, the attention of the attentive
baselines leaves out its regions, and the Sequential decoder gets no image context. An image file that cannot be
read or transformed counts as missing too: the one grayscale image of the data cannot be normalized, as in the
other pipelines. Without the option, as for the other datasets, a missing image is a blank image.

`resize_images.py` keeps the subdirectories of `images/`: the copies are `images_256/<digits>/<id>.png`. Most
photos are 1166 × 1750.

## The FashionGen data with its test set

The settings in `experiments/fashiongen_seeded/` run the 18 experiments on the challenge release of
[FashionGen](https://arxiv.org/abs/1806.08317) (`fashiongen_256_256_train.h5` and
`fashiongen_256_256_validation.h5`; its use is subject to the terms of the release), with the steps, options and
fixes of the Kaggle data, except the split: the release's validation file is a fixed test set, and the seed draws
the validation set from its training file. The image models use all the views of a product.

```bash
S=experiments/fashiongen_seeded/txt+img/sequential.json
F=/data/fashiongen   # the export: products.csv and images/
W=work/fashiongen_seeded/txt+img/sequential/seed1
python fashiongen_export.py -h5_dir /data -output $F    # once; needs h5py
python run.py $S train     -data_dir $F -work_dir $W -seed 1
python run.py $S translate -data_dir $F -work_dir $W    # with model.best.chkpt
python run.py $S evaluate  -data_dir $F -work_dir $W
```

The release has one row per photo: a product has one to six photos, one per view (`input_pose`, `id_gridfs_1` to
`id_gridfs_6`), and each of its rows repeats its text and attributes; 89 products have repeated rows, 395 in all
(their views stored twice, a few three times), with the same photos. `fashiongen_export.py` writes one row per
product to `products.csv` (the
fields of the release without the `input_` prefix, the file of the release in `split`, the views in `poses`) and
each view once to `images/<product_id>/<pose>.png` (256 × 256, saved without loss and read back). It stops if the
rows of a product disagree or a repeated view is another photo. The text is UTF-8 or, where it is not, Latin-1
(brands such as `Chloé`); it is written unchanged.

`train` (`train.py -fashiongen_dir $F -seed 1`) prepares the data in `$W/data` with `fashiongen_data.py`, from the
67,666 products of `products.csv`:

- the text: the name and the description as they are in the release, `<name>. <description>`; only a line break
  in it becomes a space (in 594 descriptions, mostly `\r\n`), since the data files hold one product per line.
  `concat_description` is not used: it also describes the other items of the photographed outfit;
- the attribute chain: the 6 attributes from coarse to fine (table below), with spaces in a value replaced by `_`;
  the release has no missing values;
- the image: one name per view slot, `<product_id>/<pose>.png`, or `None` for a view the product does not have.
  48,324 products have 4 views, 16,725 five or six, and 2,617 one to three;
- the split: the 7,519 products of the release's validation file are the test set; the seed draws 5,000
  validation products from the 60,147 of its training file, and the other 55,147 are the training set. The
  SentencePiece model, the vocabularies and `data.pt` are made as for the Kaggle data.

| # | Attribute | Field of the release | Values |
|---|---|---|---|
| 1 | department | `input_department` | 4 |
| 2 | gender | `input_gender` | 2 |
| 3 | category | `input_category` | 48 |
| 4 | subcategory | `input_subcategory` | 121 |
| 5 | season | `input_season` | 10 |
| 6 | brand | `input_brand` | 570 |

The order was chosen as for the H&M data: the category determines the department, which comes first; of the
orders that keep this rule, the prefixes of the chain have the lowest summed entropy over the products of the
release's training file. The same order results on two random halves of them and on all products; gender before
department would add 0.02 bits. 9 subcategory names stand under two categories (`SHIRTS` under `SHIRTS` and
`TOPS`, the jewellery under `JEWELRY` and `FINE JEWELRY`, ...); the category before them in the chain tells them
apart.

The image models see all views of a product (`"max_views": 6` in the settings, `train.py -max_views`). The
ResNet-50 encodes each view, and the image features are the 196 regions of every view, one view after the other,
with a learned embedding of the view slot (one per pose; zero at the start) added to its regions. The Sequential
decoder and the attentive baselines attend to the regions of the views the product has; MultiClass, Binary
MultiLabel and MultiTask average them. With `max_views` 1, the default, a product has one image, as for the other
datasets. The photos are 256 × 256 already: the resizing of `resize_random_crop` and `center_crop` (shorter
side 256) leaves them as they are.

## Settings for a fair comparison

The settings in `experiments/<dataset>/` reproduce the paper's runs, with training details that differ between
the models or from the description of the paper (Section 4.4). The settings in `experiments/<dataset>_fixed/`
(the paper's splits of Rakuten, Kaggle and Amazon) and in `experiments/kaggle_seeded/` train all the models of a
dataset in the same way, as the paper describes, without changing the models:

| Training detail | The paper's runs | The fixed settings | Option |
|---|---|---|---|
| Training crop | `RandomCrop` for some Rakuten and Amazon models, `RandomResizedCrop` for the others | Rakuten and Amazon: the random resized crop and flip that the paper describes, for all; Kaggle: resized (shorter side 256), then a random crop and flip (see below) | `-train_crop random_resized_crop`; Kaggle `-train_crop resize_random_crop` |
| Validation and test images | random crop and flip | resized and center-cropped, without randomness | `-valid_crop` and `-test_crop center_crop` |
| Model selection and early stopping | the accuracy of the training targets of each model (tokens given the gold previous ones, attributes or chains), the loss for Binary MultiLabel | the full-chain accuracy of the predicted validation chains, for all models | `-select_by chain_accuracy` |
| Pretrained ResNet | its BatchNorm statistics are updated in training | entirely frozen, as the paper describes | `-frozen_resnet` |
| Title padding in the baselines | part of the average pooling and of the attention | left out, as in the attention of the Sequential model | `-mask_padding` |
| Regularization of Vaswani et al. | dropout 0.1 | dropout 0.1 and label smoothing 0.1 | `-label_smoothing 0.1` |
| Amazon epochs | the checkpoint of the 8th epoch (`model.7.chkpt`) | 7 epochs and the last model, as the paper describes | `-epoch 7 -save_mode last` |
| Token vocabularies | the most frequent token shares the index of `<UNK>` | every token has its own index | `build_vocab.py -distinct_unk` |
| Missing attribute values (Kaggle: `NA` in `styles.csv`, `None` for an empty field; 0.8% of the products) | values that the models learn to predict and that are scored | left out of the vocabularies, the losses (index -1, `ignore_index`) and the scores; after a missing value, the Sequential decoder gets `<UNK>` as input | `"missing_values": ["NA", "None"]` (`-missing_values NA,None`) |

The Kaggle photos show the products on a white background, and the random resized crop takes 8% to 100% of a
photo: on 2,000 training photos of seed 1 (five crops each), 0.9% of its crops were at least 99% background
and 4.0% at least 90% (accessories 1.8% and 9.0%, footwear 1.7% and 4.5%). The Kaggle settings therefore
resize the photos first, as for validation and test, and take a random 224 × 224 crop and flip, which always
shows most of the photo: none of these crops was at least 99% background, 1.7% at least 90% (the center crops
of validation and test: 1.4%). The paper describes the random resized crop.

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

- `*.img`: image file name, relative to `-image_dir` (FashionGen: one name per view slot, separated by tabs,
  `None` for a view the product does not have);
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
- **The Kaggle training crop and the images resized in advance** (`-train_crop resize_random_crop`,
  `resize_images.py`): on both stacks, `tests/smoke_test.py` passes for the 24 settings with this crop (on the
  new stack for all 102 settings files) and `tests/reproducibility_test.py` for the 12 seeded image settings,
  which give the same data, weights, logs and predictions with the resized images. For all 44,441 Kaggle
  images, `center_crop` and the seeded and unseeded random crops of `resize_random_crop` give the same tensors
  from the original and from the copy. One epoch of `experiments/kaggle_seeded/txt+img/sequential.json` with
  seed 1 on a V100 with 6 CPUs gives the same data, weights, logs and test predictions with the original
  images and with the copies, in 4.0 instead of 9.9 minutes (training 3.1 instead of 7.7 minutes, the test
  predictions 47 instead of 99 seconds).
- **The H&M data** (`hm_data.py`, `experiments/hm_seeded/`, `-mask_missing_images`): on both stacks,
  `tests/smoke_test.py` passes for the 18 H&M and the 18 seeded Kaggle settings (on the new stack for all 120
  settings files), `tests/reproducibility_test.py` for these 36 settings (the 24 image settings also with the
  resized images), and `tests/masking_test.py` for the 12 image models: with all images present, the outputs
  are those of a plain batch of images, bit for bit, and the image of an absent product changes neither the
  scores nor the predictions. The 18 seeded Kaggle settings give the same data, weights, logs and predictions
  as before these changes. On the real data, seed 1 gives 95,542 training, 5,000 validation and 5,000 test
  articles, whose texts and chains are those of `articles.csv`; all 105,100 resized copies give the same tensors
  as the originals (the one grayscale image is missing in both). One epoch on a V100 with 6 CPUs trains in 17.3
  minutes for the text+image Sequential model (6.7 GB of GPU memory), 16.0 for text+image MultiClass (5.2 GB)
  and 5.0 for the image-only Attentive MultiTask model (1.2 GB).
- **The FashionGen data** (`fashiongen_export.py`, `fashiongen_data.py`, `experiments/fashiongen_seeded/`,
  `-max_views`): `tests/fashiongen_export_test.py` passes, and the export of the release has 67,666 products and
  292,623 view images, each read back equal to its photo in the HDF5 files. On both stacks,
  `tests/smoke_test.py` passes for the 18 FashionGen settings (on the new stack for all 138 settings files),
  `tests/reproducibility_test.py` for them (on the new stack also for the 36 seeded Kaggle and H&M settings), and
  `tests/masking_test.py` for the 12 image models with views: the image in the slot of an absent view changes
  neither the scores nor the predictions, a present view and its slot do, and with the view embedding at zero a
  product with one view gets the scores of the model for one image (compared in float64). The 18 seeded Kaggle
  and the 18 H&M settings give the same data, weights, logs and predictions as before these changes. On the real
  data, seed 1 gives 55,147 training, 5,000 validation and 7,519 test products (those of the release's
  validation file), whose texts, chains and views are those of the export; the resizing of `center_crop` and
  `resize_random_crop` leaves the 256 × 256 photos as they are. One epoch on a V100 with 6 CPUs trains in 12.9
  minutes for the text+image Sequential model (15.7 GB of GPU memory, on a 16 GB V100) and 5.2 for the image-only
  Attentive MultiTask model (13.8 GB), with the same data, weights, logs and test predictions on another node.

## Requirements

Python 3.6 or later, PyTorch 1.1 or later with torchvision, NumPy, Pillow and tqdm; a GPU (`-no_cuda` runs on
the CPU, slowly). Tested with Python 3.7, PyTorch 1.1.0, torchvision 0.3.0 and Pillow 6.2.2 (versions of
2019, when the experiments were run with Python 3.6 and PyTorch 1.x), and with Python 3.12, PyTorch 2.14.1,
torchvision 0.29.1 and Pillow 13.0.0, which decodes some JPEG images slightly differently (see
[Verification](#verification)). The Kaggle data split with a seed needs SentencePiece (`pip install
sentencepiece`), the FashionGen export h5py, the heat-map script OpenCV and scikit-image,
`analysis/label_stats.py frequency` matplotlib.
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
