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
8. [Contributions](#Contributions)
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
   - **SQuAD:** Squad_v2, Exact Match (EM), F1, BLEU, METEOR, ROUGE.

---

## **Performance Metrics**

### **Task 1: SST-2 (Classification)**

| Model      | Accuracy           | Precision          | Recall             | F1                 |
| ---------- | ------------------ | ------------------ | ------------------ | ------------------ |
| Zero-shot  | 0.3598099621408953 | 0.7842165895859535 | 0.3598099621408953 | 0.4854944203120602 |
| Fine-tuned | 0.9441763788879816 | 0.944726401331474  | 0.9441763788879816 | 0.9442583244134678 |

### **Task 2: SQuAD  (Question-Answering)**

| Model      | Exact Match (EM) | BLEU     | METEOR  | ROUGE-1 | ROUGE-2 |	ROUGE-L |
| ---------- | --------          | --------| ------- | -------- |------- |------- |
| Zero-shot  | 0.391                 | 0.0824838608|0.4002828852 | 0.4144738029| 0.3713187815	|0.4143814001 |
| Fine-tuned | 0.456                 | 0.4494106089  | 0.4278987724 | 0.5170069249 | 0.3576785714	|0.5170885547|

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
2. **Fine-tuned Model Links:**

   - **[OneDrive Link](https://your-onedrive-link.com)** (Download the fine-tuned model files.)
   - **[SST(Hugging Face Hub Link)](https://huggingface.co/hmhm11901/llama-3.2-sst-finetuned)** (Access the model directly on the Hugging Face Hub.)
   - **[SQUAD_v2(Hugging Face Hub Link)](https://huggingface.co/hmhm11901/llama-3.2-squad-finetuned)** (Access the model directly on the Hugging Face Hub.)


---

## **Key Insights**

<!--1. **Metrics Analysis:**

   - Higher scores post fine-tuning highlight the adaptability of large models to task-specific requirements.
   - Zero-shot performance, while decent, lags due to lack of task-specific contextual understanding. 
2. **Parameter Behavior:**

    - The number of parameters remains unchanged after fine-tuning as weights are adjusted, not the architecture. 
3. **Zero-shot vs. Fine-tuned:**

   - Fine-tuning yields better task-specific generalization and contextual accuracy.
    -->

### 2. **Understanding Parameters Between Pretraining and Fine-tuning (5 pts)**

#### **Model Parameters Without Fine-tuning (Pretraining):**

- During pretraining, models like **LLaMA-3.2** use billions of parameters (e.g., 1B or more), capturing general linguistic patterns and contextual knowledge from large-scale diverse corpora.
- Without fine-tuning, the model relies solely on these pretrained weights, which are not adapted to specific tasks such as sentiment analysis.
- **Consequently:**
- Many predictions are ambiguous, and the model often fails to map task-specific labels like "Positive" or "Negative," leading to the frequent assignment of a "None" label.

#### **Model Parameters After Fine-tuning:**

- Fine-tuning using **LoRA (Low-Rank Adaptation)** focuses on optimizing a **subset of the model's parameters** (e.g., `gate_proj`, `q_proj`, `v_proj`, etc.) that are most relevant to the downstream task.
- Instead of updating all the pretrained parameters, LoRA introduces trainable **rank-decomposed matrices** into specific layers, leaving the majority of the original model parameters unchanged. This reduces the trainable parameters by **up to 90%**, making the process computationally efficient.
- **After fine-tuning:**
- The model better distinguishes between task-specific labels (e.g., Positive or Negative) by aligning predictions with the distribution of the fine-tuning dataset.
- Predictions are task-aware and much more consistent compared to the pretrained model.

#### **Key Differences in Parameters:**

1. **Number of Parameters:**

   - **Pretraining:** The full model's parameters (e.g., billions in LLaMA-3.2) are used, but they are fixed and not task-optimized.
   - **Fine-tuning:** Only a subset of parameters (modules targeted by LoRA) is trained, while the rest remain frozen. This drastically reduces the number of trainable parameters. For instance:
   - A 1B-parameter model may only fine-tune **a few million parameters.**
2. **Parameter Changes:**

   - **Before Fine-tuning:** All parameters are pretrained and task-agnostic.
   - **After Fine-tuning:** Parameters in the targeted modules (like `gate_proj`, `q_proj`) are updated, while others remain the same.
   - **Total Model Size:** Remains unchanged because fine-tuning modifies only a small fraction of parameters without increasing the model's total parameter count.
3. **Prediction Accuracy:**

   - The fine-tuned model performs significantly better in task-specific predictions, adapting pretrained knowledge to the sentiment analysis task and reducing misclassifications like "None."

---

### 3. **Performance Differences: Zero-shot vs Fine-tuned Models (5 pts)**

#### **Zero-shot Model:**

- In the zero-shot setting, the model heavily relies on its pretraining without any task-specific fine-tuning.
- **Key Observations:**
- For nearly **50% of the samples**, the model failed to predict either a positive or negative sentiment and instead defaulted to a "None" label.
- This behavior is due to:
  - The lack of task-specific adaptation, where the model is unsure about the label distribution and nuances in the sentiment task.
  - Ambiguities in pretraining, where no direct mapping exists for many sentiment scenarios, especially for domain-specific or edge cases.

#### **Fine-tuned Model:**

- After fine-tuning, these "None" cases are reduced to a **negligible level.**
- **Key Improvements:**
- The model effectively learns to distinguish between positive and negative sentiments by adapting to the labeled dataset, aligning its predictions with task-specific requirements.
- Fine-tuning enables:
  - Better handling of ambiguous cases through exposure to task-relevant data.
  - Improved classification boundaries for sentiment, ensuring a more robust prediction mechanism.

#### **Rationale:**

- The stark contrast between the zero-shot and fine-tuned model highlights the importance of **task-specific fine-tuning.**
- By aligning the model's parameters with the distribution and nuances of the sentiment dataset:
- Fine-tuning minimizes the prediction of "None" labels.
- It ensures higher accuracy and usability in real-world applications.

## Contributions:

| Name             | Roll No. | Contribution                                                                                                                                                                       |
| ---------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Husain Malwat    | 21110117 | Worked on classification task (SST-2), calculated accuracy, precision, recall, and F1 scores. Assisted with parameter calculations, performance analysis and subjective questions. |
| Amey Rangari     | 21110177 | Worked on Question-Answering task (SQuAD), calculated squad_v2, F1, METEOR, BLEU, ROUGE, and exact-match scores. Assisted in analyzing model performance.                          |
| Netram Choudhary | 21110138 | Worked on classification task (SST-2), assisted with metrics calculation, and analyzed model performance.                                                                          |
| Vinay Goud       | 21110125 | Worked on Question-Answering task (SQuAD), calculated squad_v2, F1, METEOR, BLEU, ROUGE, and exact-match scores. Contributed to subjective analysis and performance comparison.    |
| Dhruv Patel      | 23210035 | Worked on Question-Answering task (SQuAD), calculated squad_v2, F1, METEOR, BLEU, ROUGE, and exact-match scores. Contributed to model parameter analysis.                          |

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
