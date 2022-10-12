# Rap Gender Modeling
This is an attempt to create a model that can predict the gender of a rapper from their lyrics. I have been working on [researching gender disparities in hip-hop](https://github.com/afriedman412/rcg-new) for a while, and this seemed like an interesting extension of the project that would give me some practice implementing HuggingFace models.

As of now, the model does not work!

Some of the code came from [Sentiment Analysis with BERT and Transformers by Hugging Face using PyTorch and Python](https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/), and the current classifier model came from [DistilBert in pytorch](https://www.kaggle.com/code/samson22/distilbert-in-pytorch).



## Disclaimers
1. Since my interest is in hip-hop as a traditionally male-dominated art form, the model predicts whether or not the lyrics are by a male-identifying artist. 
2. This should not be seen as a belief or endorsement of the gender binary!
3. The training data contains male, female, non-binary, gay, straight, cis and trans artists.
4. For a deep dive on how I built the training data set, check out [this repo](https://github.com/afriedman412/rap-gender-data-project).