# Tweet-Sentiment-Extraction
The task is to extract sentiment summarization from tweet texts which are denoted by sentiment type with neural,positive or negative. 

I built two_stage models to extract tweet sentiment.The first stage model is based on Word-Piece level tokens. I apply Roberta transformer with a CNN layer on the top to predict the probabilities of the start and end index for the tokens in the sequence. I then built the second stage model based on character_level tokens to futhur compute the probilities of start and end index of sentiment summarization.The implementation of the second stage model significantly impoved the jaccard score from 7.13 to 7.26.

The frameworks: tensorflow and transfomer.
Computing platform: 1 GPU 

You can also check and run my code on [kaggle](https://www.kaggle.com/emily2008/tweet-sentiment-extraction-2-stage-models)
