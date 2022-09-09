The intelligent personal assistant, is a software
agent that can perform tasks or provide services for an
individual based on verbal commands. Moreover this virtual
assistant shall use verbal inputs for analysing emotion of the
speaker, adding a psychological value to this assistant.The design
includes a feature that advises the users based on their mood
detected using speech emotion recognition.
The model extracts various audio features using Librosa
python library. Using deep learning algorithm of Convolutional
Neural Networks, Scikit-learn’s Support Vector Machine (SVM)
classifier and Multi-layer Perceptron (MLP) classifier.

Dataset used:- RAVDESS dataset i.e., Ryerson Audio-
Visual Database of Emotional Speech and Song dataset.
THe dataset contains 1440 files each of them having
a unique filename. The filename consists of a 7-part
numerical identifier These identifiers define the stimulus
characteristic of namely 8 Emotions
(01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 =
angry, 06 = fearful, 07 = disgust, 08 = surprised).
This dataset is taken from kaggle-
https://www.kaggle.com/uwrfkaggler/ravdess-emotional-
speech-audio
• Data Cleaning This step includes extract the most ap-
propriate features into a panda data frame. This technique
of extracting the features, and dealing with different kinds
of values including null values is known as data cleaning.
• Data analysis: Now that we have got our data in our
hands, we need to understand it. the dataset is analysed
with respect to the following aspects
– Emotion distribution by gender
– Variation in energy across different emotions
– Variation of relative pace of speaking (time taken to
speak certain amount of words) and power across
emotions (frequency)
