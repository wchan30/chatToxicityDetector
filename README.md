# Game Chat Toxicity Detection
Machine learning model for detecting toxic chat messages in a gaming environment using Natural Language Processing (NLP) and Scikit-learn. The model processes chat messages and classifies it either toxic or non-toxic. The model also generates confidence scores and reasoning for toxic classification.
# Analysis Approach
In order to train a dataset full of messages, we needed a way to analyze the text. Term Frequence-Inverse Document Frequency (TF-IDF) is a great method to analyze the text since it captures the importance of each word in the message. Game chats are usually short and direct in which the TF-IDF benefits from. Because the messages are short and direct, we set n-grams range to 1-3 words, making it easier for the model to detect toxicity with combinations of up to 3 words.
# Results
We were able to achieve **87% accuracy** with a dataset of 20,000 messages. Here is the breakdown:  


**Non-toxic messages**:

- Precision: **97%**

- Recall: **89%**  

- F1-score: **93%**  



**Toxic messages**:  


- Precision: **32%**  

- Recall: **68%**  

- F1-score: **43%**  

The model was great at detecting non-toxic messages with a 97% accuracy, and was able to detect most toxic messages with a 68% recall.  
Keep in mind that the dataset was imbalanced with around 18,600 non-toxic messages and 1,400 toxic messages

