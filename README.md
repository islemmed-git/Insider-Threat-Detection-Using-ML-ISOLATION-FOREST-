# Insider-Threat-Detection-Using-ML-ISOLATION-FOREST-

# Email Anomaly Detection with Isolation Forest

## Overview
This project performs anomaly detection on email metadata using the **Isolation Forest** algorithm. The goal is to detect unusual email activities based on features like the email size, sender, recipient, attachments, and timestamps. The anomalies are detected by isolating data points that deviate from the majority of the data. The results are visualized using histograms to show the distribution of anomalies in both training and test datasets.

---

## Requirements
Make sure you have the following libraries installed:
- **Python 3.x**
- **NumPy**
- **Pandas**
- **scikit-learn** (for Isolation Forest)
- **Matplotlib**

To install the required libraries, use the following command:
```bash
pip install numpy pandas scikit-learn matplotlib

## Dataset
The dataset `data.csv` contains the following email metadata:

| Column       | Description                                         |
|--------------|-----------------------------------------------------|
| `id`         | Unique identifier for each email                    |
| `date`       | Date and time the email was sent                   |
| `user`       | User who sent the email                            |
| `pc`         | Computer or machine identifier                     |
| `to`         | Recipients of the email                            |
| `cc`         | Carbon copy recipients                             |
| `bcc`        | Blind carbon copy recipients                       |
| `from`       | Sender of the email                                |
| `size`       | Size of the email in bytes                         |
| `attachments`| Number of attachments in the email                 |
| `content`    | Email body content                                 |


---

