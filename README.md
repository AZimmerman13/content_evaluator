# content_evaluator
Machine Learning Project designed to determine if a given meme (image with overlayed text) contains hateful speech.

Approach:
My approach is to train seperate models to identify whether the text or image, respectively merit being flagged.  I expect to use NLP and clustering to find latent topics in the text and then cluster the results of a CNN to group similar types of images.  I will then train a third model to take the cluster assigments of the text and image attributes of the post, and decide if the post contains hateful speech.  I expect certain image types to act as either negations or amplifications for the text portion.

Final model feature space:
NLP cluster  | image cluster | contains_slur | protected-cat | label

OR

NLP vote | CNN vote | label




in addition to featurizing each text entry as a 'document' in terms of NLP, I will also create a few common sense features to reduce the workload of the model:

* contains_slur: binary variable that identifies the use of known slurs, these should always be predicted as the positive class (contains hate speech)

* protected_cat binary variable that identifies if a group of any kind is being singled out, this will have a more complicated relationship with my labels, as one can imagine both positive and negative memes that mention groups of people.


Imagine an image with the caption "Look at all the things I love about you"
if the image itself was a barren desert landscape, the model ought to predict a different outcome than if the image showed something more positive.

## NLP

I started with a Multinomial Naive bayes model

count vectorizer accuracy: 0.76

Tfidf accuracy: 0.81 (up to 0.83 with alpha=0.05)


Random Forest (w/ Tfidf) Vectorizer:
Train accuracy: 0.8772941176470588
AUC: 0.9655263648668972

[[5109  341]
 [ 702 2348]]