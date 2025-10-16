+++
title = "More on BERT"
draft = false
[build] 
list = "never"
render = "always"
+++

## What Was BERT Pretrained On?

**BERT** was pretrained on two major English text datasets:

1. **English Wikipedia** — approximately *2.5 billion words* (plain text only — no lists, tables, or markup).

2. **BookCorpus** — around *800 million words* drawn from ~11,000 unpublished/self-published books, providing long-form narrative and conversational language.  


Because of this training, BERT was exposed to:

- A wide range of topics (science, fiction, general knowledge)
- Real English grammar and semantic variety
- Both formal (encyclopedic) and informal (narrative) writing styles

---

## Pretraining Objectives

During pretraining, BERT learns by solving two **self-supervised tasks**:

### Masked Language Modeling (MLM)

Some words in a sentence are randomly replaced with a `[MASK]` token,  
and BERT must predict the missing word using both left and right context.  
This teaches deep understanding of grammar and meaning.
 
> “The cat sat on the ___.” ... “mat”

[See More](https://aclanthology.org/2021.emnlp-main.249.pdf)

---

### Next Sentence Prediction (NSP)

BERT also learns whether one sentence logically follows another.  
This helps it understand relationships between ideas rather than just individual sentences.


> Given the sentence, "I made tea.", the next best guess would be, "Then I poured it into the mug." Rather than an unrelated sentence like, "The moon was bright last night."

## Architecture Overview

BERT is built entirely on the **Transformer encoder** architecture,  
first introduced in the paper *“Attention Is All You Need”* (Vaswani et al., 2017).

- BERT uses multiple layers of Transformer encoders  
  (Base = 12 layers, Large = 24 layers).  
- Each layer applies **self-attention**, which allows the model to understand the relationship between every word in a sentence.  
- This **bidirectional context** is what makes BERT powerful —  
  unlike earlier models such as GPT (which read left-to-right) or RNNs (which process sequentially).  

[See More](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)

## Model Variants

| Model | Layers | Hidden Size | Parameters | Notes |
|--------|---------|-------------|-------------|--------|
| **BERT-Base** | 12 | 768 | 110M | Standard version for most NLP tasks |
| **BERT-Large** | 24 | 1024 | 340M | Higher accuracy but computationally intensive |
| **DistilBERT** | 6 | 768 | 66M | Lighter, faster, distilled from BERT-Base ([See More](https://arxiv.org/abs/1910.01108)) |

Noteably, **DistilBERT**, used in my YouTube Comment Classifier project, retains roughly **97% of BERT’s accuracy** while running about **60% faster**.

---

## Fine-Tuning BERT

BERT is not designed for one specific task as it is a *general-purpose language representation model*.  
After pretraining, it can be **fine-tuned** for a variety of downstream tasks:

- Sentiment Analysis  
- Named Entity Recognition (NER)  
- Question Answering (e.g., SQuAD)  
- Text Classification  
- Natural Language Inference  

Fine-tuning usually adds a small output layer on top of BERT,  
while the pretrained weights remain mostly unchanged.

[See More](https://arxiv.org/abs/1810.04805)

---

## Tokenization

BERT uses **WordPiece tokenization**, which breaks uncommon words into smaller subword units.  
This approach lets BERT handle rare or unseen words efficiently.

**Example:**  
> “unbelievable” → `["un", "##bel", "##ievable"]`

[See More](https://www.tensorflow.org/text/guide/subwords_tokenizer)

---

## Benchmarks and Performance

Upon release, BERT achieved state-of-the-art results on multiple NLP benchmarks:

- **GLUE** – General Language Understanding Evaluation  
- **SQuAD v1.1 and v2.0** – Question Answering  
- **SWAG** – Commonsense Inference  

These results demonstrated BERT’s capability to generalize across many natural-language tasks.

---

## Limitations

While BERT remains foundational to NLP, it has several known limitations:

- **Computational cost:** Large models like BERT-Large require significant GPU/TPU resources.  
- **Context limit:** The input length is capped at 512 tokens.  
- **Non-generative:** BERT cannot generate new text; it only understands and classifies.  
