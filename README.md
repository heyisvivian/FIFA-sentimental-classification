Twitter sentiment analysis on Real-Time Scenarios
 
1 Motivation and Problem Definition
This project seeks to understand the emotional evolution, sentiments, and reactions of individuals as they express themselves through tweets during events. This in-depth analysis enables us to identify trends and gain insights into how societies´ emotional landscape evolves over a period of time.
The project aims to develop a robust sentiment analysis model for Twitter data, focusing on real-time sentiment monitoring during significant real-world events. It has a two-fold approach. First, model training, testing and validation on a pre-existing twitter dataset that consists of tweets with their corresponding sentiment. Second, creating our own unique dataset by scraping twitter for a chosen event’s tweets. The developed model is then implemented to predict sentiments in real-time tweets.
1.1 Motivation
The motivation behind this project stems from the profound impact and influence of social media in contemporary society. Social media platforms have become integral to people's lives, serving as primary outlets for emotional expression and opinion-sharing. Among these platforms, Twitter stands out as a powerful and rapid medium for real-time, text-based self-expression. It provides an unfiltered and immediate channel for public sentiments. Understanding these sentiments, especially how they change during events, offers valuable insights into people's perspectives, priorities, influencing factors, and current trends.
1.2 Problem Definition
The core challenge of this projects revolves around sentiment analysis, which entails the following key components:
1.	Sentiment Classification: Developing a machine learning model capable of classifying tweets into positive, negative, or neutral sentiments. This model must adapt to the wide range of expressions, lexicons, and nuances of online language.
2.	Real-World Event Monitoring: This aspect involves analyzing the changes in sentiment during specific real-world scenarios. The model must handle the dynamic fluctuations in public sentiments and the evolving reactions as the event unfolds.
1.2 Importance of Problem
Understanding public sentiment, particularly during real-time events, is invaluable for various stakeholders. These insights provide real-time windows into people's perspectives and emotions. A few examples: (1) monitoring and comprehending viewer sentiments during sports matches provide profound insights into human triggers, stimuli, emotions, and psychology, shedding light on the emotions dynamics of individuals. (2) political campaigns and media outlets can leverage real-time sentiment analysis to gauge public sentiment towards candidates, policies, and election outcomes. It not only improves outcome prediction but also to adapt campaign strategies based on real-time public reactions. (3) sentiment analysis offers insights into consumer perception, enabling businesses to tailor their strategies to improve customer engagement, satisfaction, and retention.
1.3 Potential Applications
1.	Sport Analysis.
2.	Election Night Analysis.
3.	Brand and product Analysis.
1.4 Related work
There are various machine learning models used in sentiment analysis, a few are as follows:
1.	Natural Language Processing (NLP).
2.	Naive Bayes Classifier.
3.	Support Vector Machines (SVM).
4.	Logistic Regression.
5.	Random Forest and Decision Trees.
2 Methodology
In this sentiment analysis methodology, we aim to provide a methodical framework for categorizing tweets as good, negative, or neutral. After preparing the data and extracting the features, we choose and train the models and finally deploy the results.
2.1 Phase 1: Data Preparation
●	Data Collection: Gather a sizable dataset of tweets with sentiment labels (positive, negative, neutral).
●	Tokenization: Split the text into words, assigning a token to each word.
●	Stop-Word Removal: Eliminate common words that may not carry sentiment information.
2.2 Phase 2: Feature Extraction
●	Bag of Words (BoW): As per the BoW model for natural language processing, create a matrix where each row represents a tweet and each column a unique word.
●	After implementing BoW, introduce either word embeddings or contextual embeddings to enhance sentiment analysis accuracy and context-awareness.
2.3 Phase 3: Model Selection
●	Identify the most suitable machine learning models for sentiment analysis, ensuring that the chosen model can effectively classify tweets into positive, negative, or neutral sentiment.
●	Consider a range of models, including Naive Bayes (in the absence of contextual embeddings), as well as more advanced options such as Support Vector Machines, Logistic Regression, and Decision Trees, to achieve optimal sentiment analysis results.
2.4 Phase 4: Model Training
●	Train the selected model on the training dataset, experimenting with parameters such as batch sizes or L1 or L2 regularization with the ultimate goal to fine-tune model parameters and avoid overfitting.
2.5 Phase 5: Model Evaluation
●	Evaluate the models on the test dataset using metrics like accuracy, precision, F1-score, and ROC-AUC.
2.6 Phase 6: Fine-Tuning
●	Based on the model's test data performance, fine-tune hyperparameters, experiment with different word embeddings, or adjust the feature extraction process.
2.7 Phase 7: Deployment
●	Implement the sentiment analysis model in a live environment, by scraping and assessing tweets pertaining to specific events, such as football matches, elections, or other noteworthy occurrences enabling a real-time evaluation of sentiment surrounding those events.
2.8 Phase 8: Documentation and Reporting
●	Provide a report documenting the entire process, including data sources, preprocessing steps, model architectures, and deployment details and outlining the findings, challenges, and solutions of our project. 
2.9 Relation with prior work	
In our project we aim to use Twitter posts for our sentiment analysis. This requires a particular methodology from, for example, feature extraction for tweets can include extracting features related to user engagement and short-text specific features while for news or articles might focus more on the textual content itself, using techniques like TF-IDF, word embeddings, or more extensive linguistic analysis.
In terms of model selection, while there are similarities with prior work in this phase, Twitter sentiment analysis models must account for the idiosyncrasies of short and noisy text. For longer articles, recurrent neural networks (RNNs) or transformer models can be effective, whereas simpler models like logistic regression may suffice for Tweets.
In terms of the deployment phase for Twitter sentiment analysis, it might involve real-time monitoring of tweets and swift responses to emerging trends. On the other hand, deployment for news sentiment analysis may involve batch processing of articles from various sources.
Adapting the methodology to suit the characteristics of the data source is essential for building accurate sentiment analysis models. Understanding the nuances of Twitter data and news/articles helps ensure the effectiveness of sentiment analysis in various real-world applications, such as social media monitoring, customer feedback analysis, and news sentiment tracking.
3 Evaluation
Expanding upon the concepts introduced in phases 4 to 7 of the methodology, the following initiatives outline a more comprehensive approach for evaluating our machine learning algorithm's effectiveness in determining sentiment from Tweets.
●	Evaluation Metrics: Beyond just accuracy, evaluating sentiment analysis involves precision, F1-Score, examining the confusion matrix, and assessing the ROC-AUC and PR-AUC curves. These metrics provide a more comprehensive view of model performance.
●	Cross-Validation: To ensure robustness, cross-validation techniques, such as k-fold cross-validation, are employed to validate the model's generalizability and to detect potential overfitting.
●	Models Comparison: Comparative analysis result between different existing models, helps us to check the authenticity and the training result of our algorithm.
●	Qualitative Evaluation: Qualitative assessment involves scrutinizing the model's outputs to understand its reasoning and identify any potential biases or misclassifications. This step delves into the interpretability and trustworthiness of the model.
●	Real-World Testing and User Review: The real-world application of sentiment analysis is where the model faces the ultimate test. Continuous monitoring and user reviews provide invaluable insights into the model's practical performance and the necessary adjustments needed for accuracy and user satisfaction.

In conclusion, evaluating a machine learning algorithm for analyzing the sentiment of an article requires a combination of quantitative and qualitative assessments, including appropriate metrics, data collection and preprocessing, cross-validation, and real-world application testing. Regularly reviewing and improving the algorithm's performance is essential to ensure that it meets the intended objectives while maintaining ethical standards and fairness.
3.1 Experimental Approach
●	Conducting Twitter sentiment analysis before and after sports matches, such as football or rugby games, involves a systematic process. By collecting and preprocessing relevant tweets, applying a sentiment analysis model, and performing statistical analyses, we can uncover how match events impact fan reactions. This real-world experiment provides valuable insights into the emotional dynamics of sports fandom and can inform decisions and strategies related to marketing, fan engagement, and event management. Additionally, ongoing validation and model refinement are crucial for maintaining accuracy and relevance in this dynamic context.

3.2 Database
We will use a database known as “Twitter Sentiment Analysis”, accessible on Kaggle via the following link: https://www.kaggle.com/competitions/twitter-sentiment-analysis2/overview.

REFERENCES
[1]	Bo Pang, Lillian Lee, & Shivakumar Vaithyanathan. (2002). Thumbs up? Sentiment Classification using Machine Learning Techniques. In Proceedings of the 2002 Conference on Empirical Methods in Natural Language Processing (EMNLP '02). ACM. DOI: https://doi.org/10.48550/arXiv.cs/0205070
[2]	Kharde, V. A., & Sonawane, Prof. S. S. 2016. Sentiment Analysis of Twitter Data: A Survey of Techniques. International Journal of Computer Applications, 139(11), 5-15. April 2016. DOI: https://doi.org/10.5120/ijca2016908625
[3]	ACM Transactions on Management Information Systems. (2018). The State-of-the-Art in Twitter Sentiment Analysis: A Review and Benchmark Evaluation. ACM Transactions on Management Information Systems (pp. 1–29). https://doi.org/10.1145/3185045
[4]	S. Asur and B. A. Huberman. 2010. Predicting the Future with Social Media. In Proceedings of the 2010 IEEE/WIC/ACM International Conference on Web Intelligence and Intelligent Agent Technology (WI-IAT ’10). ACM, New York, NY, USA, 492–499. DOI: https://doi.org/10.1109/WI-IAT.2010.63
[5]	Jemai, Word F., Hayouni, M., Baccar, S. (2021). Sentiment Analysis Using Machine Learning Algorithms. Proceedings of the 2021 International Wireless Communications and Mobile Computing (IWCMC), Harbin City, China, 775-779. DOI: https://doi.org/10.1109/IWCMC51323.2021.9498965
[6]	Azhar Yebekenova. (2017). Twitter sentiment analysis. Kaggle. https://kaggle.com/competitions/twitter-sentiment-analysis2





