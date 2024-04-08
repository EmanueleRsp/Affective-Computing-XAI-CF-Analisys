# Overview

This is my thesis work, made for bachelor thesis in Computer Engineering in Pisa.

It represents the project developed during the analysis. Code is implemented in Python, it's written in english and well commented, while thesis documentation is in italian (you can find it in [Tesi.pdf](/Tesi.pdf)). A short presentation is also available in [Presentazione.pdf](/Presentazione.pdf).

The code is based on a modified version of _cfnow_ and _counterplot_ libraries, developed to resolve some issues linked to the type of database analyzed and the task I had (_non-binary classification problem_). To look up to the main changes I made, you can refer to the **Chapter 5** of the thesis doc linked before.

To provide a quick overview, the abstract follows here (for the italian version go to the link above).

## Abstract (english version)

**Emotion recognition** is the area of AI that enables systems to identify emotions, developing the emotional intelligence needed to move within society. Many systems are designed to work alongside people: emotion recognition through AI is therefore crucial in the process of machine-human integration in order to make them more sensitive, empathetic and adaptable to the emotional needs of users. This can be particularly useful in fields such as medicine: emotion recognition systems can be used to monitor the emotional state of patients, to provide personalized care and support, or in order to identify mental disorders, enabling targeted and timely interventions. 

In these sensitive contexts, it is incumbent to be able to identify the decision-making process that leads to a given prediction, so as to check its validity. The problem with many models, however, is their structure, as this makes it difficult to identify the mechanisms underlying the decisions made, vitiating their applicability.

The study conducted is based on _K-EmoCon_, a multimodal dataset that combines the annotation of emotions from three different viewpoints with biometric measurements.

The work was initially to test some classification algorithms in order to compare their accuracies: biometric data, appropriately processed, were used by the models to make predictions about the positivity or negativity of the emotion (_valence_) felt by the subject. Specifically, models were trained based on _MLPClassifier_ and _SVClassifier_ which rendered high accuracy in the results.

Next, the work focused on using **eXplainable Artificial Intelligence** (XAI) methods in order to determine the influence and contribution of each feature in classification choices. Specifically, the method chosen was counterfactual computation, in which minimal changes in features are analyzed that allow for a different classification of the data. During these analyses, modifications were made to the library used in order to adapt it to the type of classification under study. 

Finally, the results obtained from the various models were compared to identify any similarities in the decision choices made and increase the validity of the commonalities: this provided useful information for a better understanding of the main components that influence perceived emotions during social relationships.
