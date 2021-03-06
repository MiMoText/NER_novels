# NER_novel.py

This script performs the actual NER. It uses `pandas` and `spaCy` and it accepts several configuration parameters.


## Prerequisites

This script depends on `spaCy` and `pandas`. They can be installed individually or with something like 

```bash
pip install -r requirements.txt
```

It also needs a French language model for spaCy. It can be installed by running 

```bash
python -m spacy download fr_core_news_lg
```

once inside your python environment. If you have used `pip` to install your dependencies then the language model should already have been installed.


## Basic Usage

By default, the script operates on all `.txt` files it finds at the location `../../roman18/XML-TEI/files/`. It stores its results in a file called `` in the current folder. If this is okay by you, you can simply run the script from this directory without any parameters:

```bash
python NER_novels.py
```

For a more fine-tuned usage you can either modify the configuration variables at the top of the script (see following section) or you can use equivalent command line parameters. You can list all available parameters with

```bash
python NER_novels.py --help
```

If you e.g. only want to process a single text file, you would use:

```bash
python NER_novels.py -s path/to/Voltaire_Candide.txt -r results_candide.csv
```


## Configuration

There are several config variables at the top of the script which can be adjusted as needed. These are:

- `SOURCES_PATH`: The (relative or absolute) path of the directory where the plain text input files are located. Every .txt file at this location will be used. Default is `"../../roman18/XML-TEI/files/"`. Optionally, you can specify a complete path to one specific `.txt` file, if you only want to process a single document.
- `RESULTS_PATH`: The (relative or absolute) path and the filename of the file in which the results will be stored. Default is `"../raw_results/ner_loc_per.csv"`.
- `CHUNK_SIZE`: Some of the larger files might cause your machine to run out of memory. To prevent that, we can split texts into smaller chunks with this number of words each, determine NEs separately, and add the counts back together. This might in rare cases decrease the performance of the NER, e.g. when a phrase is split across segments. Set this option to `None` to disable chunking completely, if sufficient RAM is available. Default is `120_000`.
- `MAX_DOC_LENGTH`: spaCy uses a maximum text length, given in characters, when creating its document representation. This can be used to prevent Out-Of-Memory situations, as the parser and NER models of spaCy require roughly 1GB of memory per 100000 characters in the input text. SpaCy's own default is `1_000_000`, our default is `2_800_000` which is enough to process every document we currently have. However, if `CHUNK_SIZE` is used, our documents will most probably never reach such a size.
- `RELOAD_MODEL`: If this setting is a number N, the script will reload the spaCy language model every N iterations of NER. This can save a bit of memory in cases where spaCy's internal String Store grows too large. However, this should normally not be our bottleneck, and since it increases runtime drastically, the default is `None` to disable reloading completely.
- `MOST_FREQUENT_COUNT`: How many NEs should be stored per text, i.e. if this is set to N, the script will store the N mostfrequent PER and the N most frequent LOC entities for each text. Default is `5`.

Each configuration option can also be set directly as a parameter when calling the script. To list the available parameters use
```bash
python NER_novels.py --help
```