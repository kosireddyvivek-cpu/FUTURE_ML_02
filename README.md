# Support Ticket Classification System (NLP + Machine Learning)

## Overview
Customer support teams often receive hundreds or thousands of support tickets every day through emails, forms, and helpdesk systems. Manually sorting these tickets into categories and identifying urgent issues can slow down response time.

This project builds a **Machine Learning based Support Ticket Classification System** that automatically:

- Reads customer support ticket text
- Classifies tickets into categories
- Assigns a priority level
- Helps support teams respond faster and reduce backlog

This system acts as a **decision-support tool for businesses**, improving operational efficiency and customer satisfaction.

---

## Problem Statement
In real-world support systems:

- Tickets are often **not categorized properly**
- **Urgent issues may get delayed**
- Support teams spend time **sorting instead of solving problems**

This project solves these problems by automatically classifying and prioritizing support tickets using **Natural Language Processing (NLP)** and **Machine Learning**.

---

## Dataset
Dataset used: **Customer Support Ticket Dataset**

The dataset contains support ticket information including:

- Ticket Subject
- Ticket Description
- Ticket Type (Category)
- Ticket Priority
- Customer information
- Resolution details

Total records: **8469 support tickets**

For this project, the important columns used are:

- **Ticket Subject**
- **Ticket Description**
- **Ticket Type**
- **Ticket Priority**

---

## Machine Learning Pipeline

The system follows the pipeline below:

Support Ticket Text  
→ Text Cleaning  
→ Feature Extraction (TF-IDF)  
→ Machine Learning Model  
→ Category Prediction  
→ Priority Prediction  

---

## Key Features Implemented

✔ Text cleaning  
- Lowercasing  
- Stopword removal  
- Punctuation removal  

✔ Feature extraction  
- TF-IDF Vectorization

✔ Ticket category classification

✔ Priority prediction (High / Medium / Low)

✔ Model evaluation using ML metrics

Bonus:

✔ Confusion matrix visualization  
✔ Class-wise performance analysis

---

## How Tickets Are Categorized

The ticket **subject and description** are combined into a single text input.

Example:

Ticket Subject:  
`Payment issue`

Ticket Description:  
`My payment was deducted twice`

Combined text:

`payment issue my payment was deducted twice`

The system processes the text using NLP techniques and converts it into numerical features using **TF-IDF**.

A **Logistic Regression classification model** is trained on historical ticket data to predict the **ticket category**.

Example output:

Input Ticket:  
`My payment was deducted twice`

Predicted Category:  
`Billing`

---

## How Priority Is Decided

The dataset includes a **Ticket Priority column** which contains labels such as:

- High
- Medium
- Low

A second machine learning model is trained to predict the **priority level** of new support tickets based on the ticket text.

Example:

Input Ticket:  
`Unable to login to my account`

Predicted Priority:  
`High`

This helps support teams quickly identify **urgent issues**.

---

## Model Evaluation

The model performance is evaluated using standard machine learning metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics help measure how well the model classifies support tickets and predicts priority levels.

---

## Example Prediction

Example input ticket:

`How do I reset my password?`

Output:

Category: Account  
Priority: Medium

---

## Business Impact

This system can help companies:

- Automatically route support tickets to the correct team
- Detect urgent issues earlier
- Reduce manual ticket sorting
- Improve customer response times
- Increase customer satisfaction

This makes the system useful for **SaaS companies, IT support teams, and customer service platforms**.

---

## Technologies Used

- Python
- Natural Language Processing (NLTK)
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Why This Project Matters

By building this project, I gained experience in:

- Real-world NLP pipelines
- Text preprocessing techniques
- Machine learning classification
- Operational ML use cases
- Building ML systems that support business decisions

Most beginner ML projects use toy datasets.  
This project demonstrates how machine learning can be used to **optimize real customer support workflows**.

---

## Future Improvements

Possible improvements include:

- Deep Learning models (LSTM / Transformers)
- BERT-based text classification
- Web interface using Flask
- Real-time ticket prediction API

---
