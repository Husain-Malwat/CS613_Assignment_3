# **Fine-tuning Llama3.2-1B/Gemma-2-2B for Classification and Question-Answering**

This repository contains the implementation of fine-tuning pre-trained language models (Llama3.2-1B or Gemma-2-2B-IT) for two distinct NLP tasks: **sentiment classification (SST-2)** and **question-answering (SQuAD v2)**. It also includes performance evaluation metrics before and after fine-tuning and insights into the changes in model parameters.

---

## **Table of Contents**

1. [Project Overview](#project-overview)
2. [Datasets Used](#datasets-used)
3. [Fine-tuning Process](#fine-tuning-process)
4. [Performance Metrics](#performance-metrics)
5. [Results](#results)
6. [Model Details](#model-details)
7. [Key Insights](#key-insights)
8. [How to Use](#how-to-use)
9. [Acknowledgments](#acknowledgments)

---

## **Project Overview**

The project explores fine-tuning large language models to adapt them for task-specific needs, using:

- **Sentiment classification (SST-2)**: Binary classification of movie reviews into positive or negative sentiment.
- **Question-answering (SQuAD v2)**: Extractive question-answering task with additional "no-answer" cases.

Key objectives include:

- Evaluating pre-trained (zero-shot) model performance on these tasks.
- Observing the effects of fine-tuning on metrics like accuracy, F1, BLEU, etc.
- Analyzing model parameters before and after fine-tuning.

---

## **Datasets Used**

1. **[SST-2](https://www.kaggle.com/datasets/atulanandjha/stanford-sentiment-treebank-v2-sst2):**

   - Sentiment analysis dataset with positive and negative movie reviews.
   - 80:20 train-test split with stratified sampling.
2. **[SQuAD v2](https://huggingface.co/datasets/squad_v2):**

   - Question-answering dataset with questions and corresponding answer spans or "no-answer" cases.
   - 80:20 train-test split using random sampling.

---

## **Fine-tuning Process**

1. **Model Selection:**

   - Pre-trained Llama3.2-1B or Gemma-2-2B models.
2. **Implementation:**

   - Used Hugging Face `transformers` library for training and evaluation.
   <!-- - Configured:
     - Optimizer: AdamW
     - Batch size: 16
     - Learning rate: 3e-5
     - Epochs: 5
     - Early stopping with patience of 2 epochs. -->
3. **Metrics Calculated:**

   - **SST-2:** Accuracy, Precision, Recall, F1.
   - **SQuAD v2:** Exact Match (EM), F1, BLEU, METEOR, ROUGE.

---

## **Performance Metrics**

### **Task 1: SST-2 (Classification)**

| Model      | Accuracy | Precision | Recall | F1 |
| ---------- | -------- | --------- | ------ | -- |
| Zero-shot  | %        | %         | %      | %  |
| Fine-tuned | %        | %         | %      | %  |

### **Task 2: SQuAD v2 (Question-Answering)**

| Model      | Exact Match (EM) | F1 | BLEU | METEOR | ROUGE |
| ---------- | ---------------- | -- | ---- | ------ | ----- |
| Zero-shot  | %                | %  | %    | %      | %     |
| Fine-tuned | %                | %  | %    | %      | %     |

---

## **Results**

Fine-tuning significantly improved performance on both tasks:

<!-- - SST-2 showed a **13.3% increase in F1**.
- SQuAD v2 exhibited a **33% boost in F1** and improvements across all other metrics. -->

---

## **Model Details**

1. **Parameter Count:**

   - Pre-trained: 1 billion (Llama3.2-1B).
   - Post fine-tuning: Parameter count remains unchanged (weights updated but model architecture intact).
2. **Fine-tuned Model Upload:**

   - [Fine-tuned SST-2 Model](#)
   - [Fine-tuned SQuAD v2 Model](#)

---

## **Key Insights**

1. **Metrics Analysis:**

   <!-- - Higher scores post fine-tuning highlight the adaptability of large models to task-specific requirements.
   - Zero-shot performance, while decent, lags due to lack of task-specific contextual understanding. -->
2. **Parameter Behavior:**

   <!-- - The number of parameters remains unchanged after fine-tuning as weights are adjusted, not the architecture. -->
3. **Zero-shot vs. Fine-tuned:**

   - Fine-tuning yields better task-specific generalization and contextual accuracy.

---

<!-- ## **How to Use**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/finetune-llm
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run fine-tuning:
   ```bash
   python fine_tune.py --model llama3.2-1b --task sst2
   python fine_tune.py --model llama3.2-1b --task squad2
   ```
4. Evaluate model:
   ```bash
   python evaluate.py --model_path ./fine_tuned_model --task sst2
   ``` -->

---

## **Acknowledgments**

- **Datasets:** Stanford Sentiment Treebank, SQuAD v2.
- **Libraries:** Hugging Face `transformers`, `datasets`, `evaluate`.

For any questions, feel free to reach out via issues or email.
