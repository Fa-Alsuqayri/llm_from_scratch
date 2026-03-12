# Build GPT from Scratch: Saudi History & Dialect Model
**Colab notebook with output:** https://colab.research.google.com/drive/1iKeJjP7pI83b6mcTWkNrUe-JDyaYTyeP?usp=sharing
**Demo:** https://youtu.be/4z6d1kvK4Nc


**Author:** Faisal Alsuqayri 

## Project Overview
This project implements a complete lifecycle of a Large Language Model (LLM) from scratch using PyTorch. The goal is to build, pretrain, and fine-tune a Decoder-Only Transformer architecture (124M parameters) customized to understand Arabic text, specifically focusing on the Saudi dialect and regional history (Najd, Hejaz, and the Islamic Middle Ages).

## Model Architecture
* **Tokenization:** Byte-Pair Encoding (BPE) using `tiktoken`.
* **Components:** Custom implementations of Multi-Head Self-Attention, Feed-Forward Networks (MLP), Layer Normalization, and Residual Connections.
* **Size:** 124 Million Parameters.

## Project Phases

### Phase 1: Pretraining (Base Model)
* **Goal:** Teach the model the fundamental structure of the Arabic language via Next-Token Prediction.
* **Dataset:** A raw text corpus extracted from historical texts, classical poetry, and regional narratives.
* **Size:** ~117000 characters.
* **Outcome:** The model successfully learned to generate Arabic text strings, heavily influenced by the historical domain of the training data.

### Phase 2: Supervised Fine-Tuning (SFT)
* **Goal:** Teach the model to follow instructions and complete tasks in a Q&A format.
* **Dataset:** Custom JSON formatted instruction-response pairs (`instruction`, `input`, `output`). 
* **Data Alignment Strategy:** Initial attempts with general conversation datasets led to severe hallucinations. To achieve optimal performance, the SFT dataset was strategically aligned with the pretraining domain, utilizing over 100+ high-quality QA pairs focused exclusively on the history of Saudi Arabia, Hejaz, Najd, and major Islamic historical events.

##Technical Limitations & Failure Modes
As part of the evaluation process, several technical limitations and failure modes were identified:

1. **Token Fragmentation (BPE Issue):** Because the model relies on `tiktoken` (optimized for English), Arabic characters are heavily fragmented into sub-bytes. This occasionally results in the model generating misspelled or gibberish words (e.g., generating "تخليج" instead of "الخليجية") as it struggles to stitch the bytes correctly.
2. **Domain Mismatch & Hallucination:**
   When prompted with topics outside its pretraining domain (e.g., modern sports), the model hallucinates by connecting the prompt to historical concepts (e.g., mentioning "رحلة الحج" instead of answering about a football match).
3. **Repetition Loops:**
   Using simple `Greedy Decoding` (`torch.argmax`) caused the small model to get stuck in infinite repetition loops. This was mitigated during inference by implementing `Top-K` and `Temperature` sampling to inject variance.

## Future Work
To transition this model from a PoC to a production-ready Arabic LLM, the following steps are recommended:
* Replace `tiktoken` with a native Arabic tokenizer (e.g., `CAMeL Tools`) to reduce token sequence length and fix fragmentation.
* Scale the pretraining dataset to 5GB+ of diverse Arabic text.
* Train entirely from scratch using randomized weights instead of adapting English-based structural weights.

