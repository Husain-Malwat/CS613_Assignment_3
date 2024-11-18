# **CS 613: NLP**

## *Assignment 3: Fine Tuning & Evaluation*

| Total marks: 100 Pts.  | Submission deadline: 23:59:59 Hrs, November 20, 2024  |

| :---- | :---- |

## **Assignment Instructions**

1. The assignment deadline cannot be extended. A 100% penalty will be incurred.
2. We will follow the zero plagiarism policy, and any act of plagiarism will result in a zero score for the assignment.
3. Please cite and mention others' work and give credit wherever possible.
4. If you seek help and discuss it with the stakeholders or individuals, please ask their permission to mention it in the report/submission.
5. Compute requirement: Use Colab and write the answers in the Colab itself.

## **Problem Statement (100 Points)**

Wherein you gave them a popular pre-trained model (say Llama3/Gemma) and asked them to finetune it for two or three different tasks. Ask them to report performance scores before and after finetuning.

**Task 1:**

1. Select the [Llama3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)/[Gemma](https://huggingface.co/google/gemma-2-2b-it) model.
2. Calculate the [**number of parameters**](https://docs.google.com/document/d/1WSiqdhmYx2Stm3JyZWgYSvj0t2-BTGabYsS74af0394/edit?usp=sharing) of the selected model from the [code](https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model). Do your calculated parameters match with the parameters reported in the respective papers of the selected model? **\[05 pts\]**
3. Fine-tune the pre-trained model on the following two tasks:
4. Classification: [SST-2](https://www.kaggle.com/datasets/atulanandjha/stanford-sentiment-treebank-v2-sst2) **\[12.5 pts\]**
5. Question-Answering: [SQuAD](https://huggingface.co/datasets/squad_v2) **\[12.5 pts\]**

*The train-test split should be 80:20, use [random/stratify](https://discuss.huggingface.co/t/how-to-split-hugging-face-dataset-to-train-and-test/20885/3) sampling and seed as 1\. Fine-tuning should be performed on the Train split.*

4. Calculate the scores for the following metrics on the test splits for the pre-trained (zero-shot) and fine-tuned models. Note that metrics depend on the selected task:

1.**Classification:** Accuracy, Precision, Recall, F1 ([Reference](https://huggingface.co/docs/evaluate/en/choosing_a_metric)) **\[20 pts\]**

2.**Question-Answering:** squad\_v2, F1, METEOR, BLEU, ROUGE, exact-match \[[Read this\!](https://huggingface.co/tasks/question-answering), and [this too](https://anthonywchen.github.io/Papers/evaluatingqa/mrqa_slides.pdf)\!, lastly [this](https://huggingface.co/docs/datasets/v1.1.0/loading_metrics.html)\!\]**\[20 pts\]**

5. Calculate the number of parameters in the model after fine-tuning. Does it remain the same as the pre-trained model? **\[05 pts\]**
6. Push the fine-tuned model to 🤗. **\[05 pts\]**
7. Write appropriate comments and rationale behind:
8. Lower or higher scores in the metrics.  **\[10 pts\]**
9. Understanding from the number of parameters between pretraining and fine-tuning of the model. **\[05 pts\]**
10. Performance difference for the zero-shot and fine-tuned models. **\[05 pts\]**

**Total  \= 05+12.5+12.5+20+20+05+05+10+05+05 \= 100 Pts.**

## **Submission**

1. Submit your code (GitHub) or colab notebook with proper comments to [this link](https://docs.google.com/forms/d/e/1FAIpQLScqZK08h4ngBm2VwfuSaoCPUKH3zxGVypNh5i3PIJNyHnENPQ/viewform?usp=sf_link).
2. Ensure the individual contribution is appropriately added (OTHERWISE PENALTY OF 10 MARKS).

Expectations from the team:

1. Properly divide the team into sub-groups and distribute your tasks equally.
2. Negative marks for not creating the report for the assignment (This can be a simple summary report).
3. Write the contributions or tasks completed by each team member. Scores might be different among team members if the tasks are not equally distributed.

## **TAs to Contact**

1. Himanshu Beniwal (himanshubeniwal@iitgn.ac.in)
2. Indrayudh Mandal (24210041@iitgn.ac.in)
3. Mithlesh Singla (24210063@iitgn.ac.in)
4. Alay Patel (alay.patel@iitgn.ac.in)
5. Aamod Thakur (aamod.thakur@iitgn.ac.in)

---

**FAQs**

1.*We will add clarifications to doubts here. Please check periodically, as someone might have already asked about the doubt, which will be appended here.*
