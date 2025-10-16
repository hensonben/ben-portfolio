+++
title = "YouTube Comment Classifier"
+++

<hr>

## 🧠 Try It Yourself

<iframe
	src="https://azande7-yt-sentiment-demo.hf.space"
	frameborder="0"
	width="850"
	height="450"
  scrolling="no"
></iframe>

## Example Prompts: (Encouraged to try your own!)
**Easy** (Clear Sentiment)  
These are simple and direct.
- "This video was amazing! Learned so much ❤️”
- “Bro this deserves way more views.”
- “The editing and pacing were perfect!”
- “This was a complete waste of time.”
- “Why did I even click on this?”
- “Terrible audio, couldn’t even finish watching.”
- "This video is 10 minutes long.”
- “Uploaded on October 16th.”
- “I came here after seeing the thumbnail.”

**Moderate** (Emotional Tone or Slightly Ambiguous)  
These require the model to pay attention to emotion and context.  
- “Didn’t expect to like it, but this was actually really good.”
- “You can tell they put real effort into this.”
- “I mean… it’s fine, but definitely not great.”
- “Could have been better. Not what I hoped for.”
- “I’m not sure how to feel about this.”
- “The topic is interesting but poorly explained.”

**Tricky** (Sarcasm, Mixed Emotions, or Subtle Negativity)  
These test how well the model understands nuance. 
- “Oh wow, another completely original reaction video 🙄.”
- “Just what the internet needed… more unboxing videos.”
- “Good visuals, but the message didn’t land.”
- “Loved the first half, but it totally fell apart at the end.”
- “Not really my thing, but I can see why people like it.”
- “Nice effort, though the execution could use some work.”

**Difficult** (Irony, Context, or Complex Sentiment)  
These are the toughest! 
- “Yeah, because *that’s* definitely how physics works.”
- “Great, another expert telling me how to live my life.”
- “This aged like milk.”
- “Can’t believe this is still relevant in 2025.”
- “I’m happy for them, but man this video made me sad.”
- "It’s so bad it’s actually kind of funny.”

## Overview

This project fine-tunes a **DistilBERT Transformer**model to classify YouTube comments into positive, neutral, or negative sentiments.
The goal was to gain hands-on experience with **Natural Language Processing (NLP)** and explore the process of fine-tuning pretrained language models using **Hugging Face Transformers** and **PyTorch**.

The dataset was sourced from Kaggle and consists of real YouTube user comments labeled by sentiment. *See end of page for a link to the dataset if so inclined.*


### Tools and Libraries
- Python 3.13
- Hugging Face Transformers (v4.57.1)
- Datasets (Hugging Face)
- PyTorch 2.8.0
- Accelerate 1.10.1

### Model and Training Details
After preprocessing, the text was **tokenized** and converted into numerical input for the model. Additionally, sentiment labels were encoded as integers (0 = negative, 1 = neutral, 2 = positive)

### Model: `distilbert-base-uncased`    
This is a lightweight version of **BERT** (Bidirectional Encoder Representations from Transformers) that retains most of BERT's accuracy while being both faster and smaller. Additionally, it's pretrained on large text data and fine-tuned here for sentiment classification. This project allowed the model to learn, rather than strictly memorize, the nuance and emotion (think sarcasm, humor, etc.) behind words in a given sentence.   
[More on BERT](/projects/yt-classifier/_bert-info/)

### Task: *Sequence Classification* 
The model learns to assign one of three sentiment categories (positive, neutral, or negative) to each input text sequence (YouTube comment).

### Train/Test Split: *80/20*   
80% of the dataset is used for training the model, while 20% is reserved for evaluating how well it generalizes to unseen data.

### Batch Size: *8* 
During training, the model processes 8 examples at a time before updating its weights. Smaller batch sizes can help with limited GPU memory and improve generalization slightly.

### Learning Rate: `2e-5`  
Controls how much the model’s weights are adjusted during training.  
A smaller learning rate helps prevent the model from “overshooting” the optimal solution.

### Epochs: *2*
An *epoch* represents one complete pass through the entire training dataset.  
After two passes, the model’s performance had plateaued, indicating that additional training would not meaningfully improve results.

### Weight Decay: *0.01*
A regularization technique that prevents overfitting by slightly penalizing large weights during optimization. This encourages the model to maintain simpler, more generalizable parameter values.

### Optimizer: *AdamW*  
A variant of the Adam optimizer that integrates weight decay directly into the update rule.  
It is widely used for fine-tuning Transformer-based models such as BERT and DistilBERT, providing stable convergence and reduced overfitting.   
[More on Adam](/projects/yt-classifier/_adam-info/)

### Results
| Metric | Score |
|---------|:------:|
| **Training Loss** | 0.37 |
| **Evaluation Loss** | 0.49 |
| **Accuracy** | 0.84 |
| **Weighted F1 Score** | 0.84 |

These metrics evaluate how well the fine-tuned model performs in classifying the sentiment of YouTube comments.

- ### **Training Loss — 0.37**  
  Represents how far the model’s predictions are from the correct labels during training.  
  A low training loss indicates that the model successfully learned from the dataset without overfitting.

- ### **Evaluation Loss — 0.49**  
  Calculated on a separate validation set that the model did not see during training.  
  The slightly higher loss compared to training is expected and shows that the model generalizes reasonably well to unseen data.

- ### **Accuracy — 0.84**  
  Indicates that the model correctly classifies about **84%** of comments into the appropriate sentiment categories (positive, negative, or neutral).  
  Accuracy provides a straightforward measure of overall correctness.

- ### **Weighted F1 Score — 0.84**  
  Balances **precision** (how many predicted sentiments were correct) and **recall** (how many actual sentiments were detected).  
  The *weighted* version accounts for differences in class sizes, ensuring that all sentiment categories are evaluated fairly.  
  A score of **0.84** demonstrates that the model performs consistently across different comment types.

---
Overall the model achieved strong performance on unseen YouTube comments, which demonstrated effective transfer learning from pretrained weights to real-world text.

### Model Access
You can test and download the fine-tuned model directly on Hugging Face:  
https://huggingface.co/azande7/yt-sentiment-model

Link to the dataset: (https://www.kaggle.com/datasets/atifaliak/youtube-comments-dataset)

---
