# PyTorch BERT Document Classification
Enriching BERT with Knowledge Graph Embedding for Document Classification (PyTorch)

Content:
- CLI script
- author embeddings
  - projector files
  - create file
- data preparation
- requirements
- trained model weights as release zips

## Installation

Requirements:
- Python 3.6
- CUDA GPU
- Jupyter Notebook

Install dependencies:
```
pip install -r requirements.txt
```

## Prepare data

### GermEval data

- Download from shared-task website: [here](https://competitions.codalab.org/competitions/20139)
- Run all steps in Jupyter Notebook: [germeval-data.ipynb](#)

### Author Embeddings

- [Download pre-trained Wikidata embedding (30GB): Facebook PyTorch-BigGraph](https://github.com/facebookresearch/PyTorch-BigGraph#pre-trained-embeddings)
- [Download WikiMapper index files (de+en)](https://github.com/jcklie/wikimapper#precomputed-indices)

```
python wikidata_for_authors.py run ~/datasets/wikidata/index_enwiki-20190420.db \
    ~/datasets/wikidata/index_dewiki-20190420.db \
    ~/datasets/wikidata/torchbiggraph/wikidata_translation_v1.tsv.gz \
    ~/notebooks/bert-text-classification/authors.pickle \
    ~/notebooks/bert-text-classification/author2embedding.pickle

# OPTIONAL: Projector format
python wikidata_for_authors.py convert_for_projector \
    ~/notebooks/bert-text-classification/author2embedding.pickle
    extras/author2embedding.projector.tsv \
    extras/author2embedding.projector_meta.tsv

```


## Reproduce paper results


Download pre-trained models: [GitHub releases](https://github.com/malteos/pytorch-bert-document-classification/releases)


### Available experiment settings

Detailed settings for each experiment can found in `cli.py`.

```
task-a__bert-german_full
task-a__bert-german_manual_no-embedding
task-a__bert-german_no-manual_embedding
task-a__bert-german_text-only
task-a__author-only
task-a__bert-multilingual_text-only

task-b__bert-german_full
task-b__bert-german_manual_no-embedding
task-b__bert-german_no-manual_embedding
task-b__bert-german_text-only
task-b__author-only
task-b__bert-multilingual_text-only
```

### Enviroment variables

- `TRAIN_DF_PATH`: Path to Pandas Dataframe (pickle)
- `GPU_ID`: Run experiments on this GPU (used for `CUDA_VISIBLE_DEVICES`)
- `OUTPUT_DIR`: Directory to store experiment output
- `EXTRAS_DIR`: Directory where author embeddings and [gender data](https://data.world/howarder/gender-by-name) is located
- `BERT_MODELS_DIR`: Directory where pre-trained BERT models are located 

### Validation set

```
python cli.py run_on_val <name> $GPU_ID $EXTRAS_DIR $TRAIN_DF_PATH $VAL_DF_PATH $OUTPUT_DIR --epochs 5
```

### Test set

```
python cli.py run_on_test <name> $GPU_ID $EXTRAS_DIR $FULL_DF_PATH $TEST_DF_PATH $OUTPUT_DIR --epochs 5
```

### Evaluation

The scores from the result table can be reproduced with the `evaluation.ipynb` notebook.

## How to cite

If you are using our code, please cite our paper:
```
@article{,
  title={},
  author={},
  journal={arXiv preprint arXiv:},
  year={2019}
}

```

## References

- [Google BERT Tensorflow](https://github.com/google-research/bert)
- [Huggingface PyTorch Transformer](https://github.com/huggingface/pytorch-transformers)
- [Deepset AI - BERT-german](https://deepset.ai/german-bert)
- [Facebook PyTorch BigGraph](https://github.com/facebookresearch/PyTorch-BigGraph)

## License

MIT


