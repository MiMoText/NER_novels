# NER novels

Perform Named Entity Recognition (NER) on french novels from the roman18 corpus with the help of SpaCy.


## Description

The roman18-corpus of about 100 eighteenth-century French novels in full text is analysed via SpaCy’s named entity recognition. Named entity recognition (NER) is a popular information retrieval technique “to identify and segment named entities and classify or categorize them under various predefined classes” (Sarkar, 2019).

![Named entity recognition](https://github.com/MiMoText/NER_novels/blob/main/img/ner_diderot.PNG?raw=true)

Within the French language package of SpaCy one can distinguish the following types of named entities: LOC, PER, MISC and ORG entities. The five most common “LOC” (location) entities within each novel and their numerical occurences per text are extracted.


### Structure of this repository

The script which performs the NER and its technical documentation is under `scripts/`. The latest results of this script are stored under `raw_results/`, although this is configurable. These results still need manual corrections and supplements to be suitable for import into the [MiMoText Wikibase](https://github.com/MiMoText/roman18). The latest version of this is under `edited_for_import/`. In this folder there is also a file named `edits_openrefine.txt`. This has been generated by [OpenRefine](https://docs.openrefine.org/) which is used for semi-automatic reconciliation and includes all the editing steps in this tool.

![OpenRefine](https://github.com/MiMoText/NER_novels/blob/main/img/OpenRefine.PNG?raw=true)


## Licence

Software in this repo, unless specified otherwise, is made available under the MIT license. We don’t claim any copyright or other rights on the metadata. If you use our scripts or results, for example in research or teaching, please reference this repository using the citation suggestion below.


## Citation suggestion

NER on Eighteenth-Century French Novels (1750-1800), edited by Julia Röttgermann, Johanna Konstanciak and Henning Gebhard. Trier: TCDH, 2022. URL: https://github.com/MiMoText/NER_novels.
