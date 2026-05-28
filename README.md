# Interactive Machine Translation

This repository implements training, evaluation, and simulation tools for interactive machine translation (IMT) using both encoder-decoder and decoder-only large language models.

The main workflow is:
1. Translate a source sentence to the target language.
2. Simulate a human interaction with the output using prefix-based or segment-based corrections.
3. Restrict future model generations according to validated segments or corrected prefixes.
4. Measure IMT costs such as word-stroke ratio (WSR), mouse-action ratio (MAR), and generation iterations.

## Repository structure

- `scr/`
  - `train.py` - train translation models with PyTorch Lightning and optional LoRA.
  - `imt.py` - run simulated interactive machine translation sessions.
  - `bleu_ter.py` - evaluate translation outputs with BLEU and TER metrics.
  - `model.py` - helper functions for model loading, dataset handling, and prompt formatting.
  - `restriction.py` - prefix and segment restriction logic for IMT simulation.
  - `unwrapp.py` - export trained Lightning checkpoints to Hugging Face-compatible model weights.
- `utils/`
  - `bleu_ter.py`, `get_means.py` - additional evaluation utilities.
  - Notebooks for analysis: `longitudes.ipynb`, `statistical_tests.ipynb`, `tabla_oficial.py`, `unks.ipynb`.
- `art/`
  - Scripts and research utilities for alternative evaluation and analysis.
- `resultados/`
  - Example output files with metrics and evaluation results.
- `environment.yaml`
  - Conda environment specification for dependencies.
- `launcher.sh`
  - Example commands for training, evaluation, and IMT simulation.

## Core functionality

### Training
Use `scr/train.py` to fine-tune models for translation tasks. The script supports:
- Source and target language selection (`-src`, `-trg`).
- Dataset folder selection (`-dir`).
- Model selection (`-model`, e.g. `mbart`, `m2m`, `flant5`, `nllb`, `llama`, `qwen`, `eurollm`, `gemma`).
- Optional LoRA adaptation (`-lora`).
- Batch size, learning rate, and number of epochs.

Example:
```bash
python scr/train.py -src en -trg gl -dir data/gl-en/ -model gemma -lora -bs 8 -e 3
```

### Model export
Use `scr/unwrapp.py` to unwrap a PyTorch Lightning checkpoint and save the underlying Hugging Face model weights.

Example:
```bash
python scr/unwrapp.py path/to/checkpoint.ckpt -src en -trg gl -model gemma -lora
```

### Standard MT evaluation
Use `scr/bleu_ter.py` to generate translations and compute BLEU and TER scores over a dataset partition.

Example:
```bash
python scr/bleu_ter.py -src en -trg gl -dir data/gl-en/ -p test -model path/to/model -model_name gemma -b 16
```

### Interactive MT simulation
Use `scr/imt.py` to simulate an IMT session and compare prefix-based vs segment-based approaches.

Example:
```bash
python scr/imt.py -src en -trg gl -dir data/gl-en/ -model path/to/model -model_name gemma -p test -out imt_output -seg
```

Options include:
- `-seg`, `--segment_based` to enable segment-based interaction.
- `-ini`, `--initial` to skip initial lines.
- `-fin`, `--final` to limit evaluation to a portion of the dataset.
- `-v`, `--verbose` for debugging output.

### Supported model families
The repository currently supports the following model families via `scr/model.py`:
- `mbart` (`facebook/mbart-large-50-many-to-many-mmt`)
- `m2m` (`facebook/m2m100_418M`)
- `flant5` (`google/flan-t5-base`)
- `nllb` (`facebook/nllb-200-distilled-600M`)
- `llama` (`meta-llama/Llama-3.2-1B-Instruct`)
- `qwen` (`Qwen/Qwen2.5-VL-7B-Instruct`)
- `eurollm` (`utter-project/EuroLLM-1.7B`)
- `gemma` (`google/gemma-3-4b-it`)

## Data layout

Datasets are expected in `folder/{partition}.{lang}` format, where `partition` is `train`, `dev`, or `test`, and `lang` is the two-letter language code. Example:
- `data/gl-en/train.gl`
- `data/gl-en/train.en`
- `data/gl-en/test.gl`
- `data/gl-en/test.en`

## Notes

- The project uses language-specific prompt formatting for decoder-only models (`llama`, `qwen`, `eurollm`, `gemma`).
- `scr/restriction.py` contains the IMT interaction logic that enforces prefix or segment corrections during generation.
- `launcher.sh` includes ready-to-use example commands for training, BLEU/TER evaluation, and IMT simulations.

## Citation
If you use this repository, please cite:
```bibtex
@inproceedings{gomez2024interactive,
  title={Interactive Machine Translation with Large Language Models in Low Resources Languages},
  author={G{\'o}mez, Sergio and Domingo, Miguel and Casacuberta, Francisco},
  booktitle={Proc. IberSPEECH 2024},
  pages={66--70},
  year={2024}
}
```
