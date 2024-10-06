# NLP-Essay-Grading-with-LSTM-BERT-and-RoBERTa
This repository contains implementations of various machine learning models used to grade essays based on six target features: cohesion, syntax, vocabulary, phraseology, grammar, and conventions.


### 1. BERT Model:
**BERT for Essay Grading**
- Implementation of a BERT-based regression model to predict six target features: cohesion, syntax, vocabulary, phraseology, grammar, and conventions.
- Custom dataset class created to preprocess the input data using a BERT tokenizer and prepare it for model input.
- Fine-tuning BERT with a regression head that outputs scores for the six features using MSE (Mean Squared Error) as the loss function.
- Hyperparameters include batch size and learning rate, with training conducted over multiple epochs.
- Plots and logs showing MSE and RMSE (Root Mean Squared Error) during training.

### 2. Classic Models Notebook:
**Traditional Machine Learning Models for Essay Grading**
- Implements classic machine learning models such as Support Vector Machines (SVM), Random Forest, and Gradient Boosting for essay scoring.
- Uses feature extraction methods such as TF-IDF to convert text into numerical features for traditional models.
- Hyperparameter tuning and cross-validation to improve model performance.
- Compares the performance of traditional models with deep learning models like BERT and LSTM.

### 3. LSTM Model:
**LSTM for Essay Grading**
- Implements an LSTM-based model for regression tasks to predict essay scores.
- Preprocesses input data using tokenization and padding for sequences, preparing input for the LSTM network.
- Uses embedding layers, LSTM layers, and a final regression head to output six target scores.
- Tracks MSE and RMSE during training and visualizes performance.
- Compares the effectiveness of LSTM in capturing the sequential nature of essay data versus other models.

### 4. RoBERTa Model:
**RoBERTa for Essay Grading**
- Implements a RoBERTa-based regression model for essay grading, similar to BERT but utilizing the RoBERTa architecture for improved performance.
- Preprocessing includes tokenization using the RoBERTa tokenizer and feeding the input to a regression head for the six target features.
- Training loop with Adam optimizer and MSE loss function, tracking loss, MSE, and RMSE.
- Performance metrics are visualized through plots for MSE and RMSE, and model performance is saved after each epoch.

## Essay Grading Data Columns

| **Column**      | **Description**                                                                        |
|-----------------|----------------------------------------------------------------------------------------|
| `text_id`       | A unique identifier for each essay                                                     |
| `full_text`     | The complete text of the student's essay                                               |
| `cohesion`      | The score representing the logical flow and cohesion of the essay                       |
| `syntax`        | The score representing the syntax and sentence structure                                |
| `vocabulary`    | The score representing the use of vocabulary throughout the essay                       |
| `phraseology`   | The score representing the appropriate use of phrases and idioms                        |
| `grammar`       | The score representing the grammatical accuracy of the essay                            |
| `conventions`   | The score representing the adherence to conventions such as spelling and punctuation    |

