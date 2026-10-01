# EP46 — Text Classification with Transformers

**Deep Learning Series · Module 9 (NLP)**

Build a production-grade text classifier by fine-tuning DistilBERT in under 100 lines.

- Binary / multi-class / multi-label
- Hugging Face pipeline (1-line classifier)
- DistilBERT architecture + classification head
- Datasets, 3-way split, dynamic padding
- Training loop + warmup + linear decay
- Confusion matrix, precision, recall, macro F1
- Production habits: batching, truncation, confidence threshold

## Files

| File | Description |
|------|-------------|
| `generate_dl_ep46.py` | Full production pipeline (TTS + Manim + merge) |
| `manim_dl_ep46.py` | All Manim scenes (clock-synced) |
| `ep46_text_classifier.py` | **The 92-line deliverable** |
| `output/` | Rendered video, thumbnail, chapters, voice files |

## Quick Start

```bash
pip install "transformers>=4.46" datasets torch scikit-learn accelerate

# Fine-tune on IMDB (takes a few minutes on GPU / ~15-20 min on CPU)
python ep46_text_classifier.py

# Predict with the saved model
python ep46_text_classifier.py "loved every minute of it"
python ep46_text_classifier.py "WIN a free prize, click now"
