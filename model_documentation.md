## Model details
This documentation includes the thought processing for model, data processing, metrics, memory and runtime, weakness and future work.

## Model 
Multi-nominal Naive Bayes Classifier is selected for below reasons:

1) Based on TF-IDF calculation, highly frequent tokens for each category are oberseved. This makes Multi-nominal NB suitable as the distribution of tokens for each category is applicable to Multi-nominal distribution. 

2) The order of tokens in a term (such as 'used car' Vs 'car used') does not impact the category of the term. 

3) Multi-nominal NB requires much less resources for train and test, compared to Gradient Boost Classifier as an example.

## Preprocessing
Preprocessing steps include:
1) Punctuation: punctuation is investigated and results show a small amount of punctuations (5) appear in some terms. Here an aggressive removal methos is taken to remove all punctuation except period sign (.). 

2) Stopwords: a customized stopwrods list is used for stop words removal. Assume that removing stop words would not impact the category of a term, becuase stopwords carry no semantic meaning. For example, 'map of cornwall uk' is converted to 'map cornwall uk'. The latter is believed to remain in the same category. 

3rd parties stopwords lists (i.e, nltk and spacy) are available. 

3) Lemmatization: lemmatization with simple POS tag is performed. Preliminary review on terms found words such as 'card', 'cards'. These same base form of words will impact TF-IDF score and TF-IDF vectorizer, without lemmatization. But note that some words may be over-chopped, such as 'used' -> 'use'. 

4) Data enrichment: Exploratory analysis found some tokens appears repeatively in a same category search, making these tokens powerful for prediction. 

In this case, TF-IDF socre for every token of each category is calculated. Random sampling of tokens with more than average of TF-IDF scores is appended to each term. In addition, assume that the informative tokens for a category are mainly Noun and Verb. 

Orignial trainSet data is first split into train and evaluation subsets. Train subset is enriched. evaluation subset is held out for evaluation.
Note train subset is further divided into train and validation set during model training. The model trian metrics data is from this train sub-subset, while model evaluation metrics is from evaluation subset.

Data enrichment may result in model overfitting, also comes at a cost of both memory and runtime.

## Metrics
Macro-precision is selected to measure the overall model performance, with the following reasoning:
1) Category imbalance analysis found no strong class imblances for 1419 categories. Majority of categories have terms between 300 to 600.

2) For minority categories, it is assumed that they have no significant business importance, as compared to majority categories. Term categories are equally important for business in each and every sectors where its clients operate. 

3) From the business perspective, precision is more relevant to live prediction scenarios. 

## Memory & Runtime 

1) Model training takes 259.0 s.
1) Model evaluation takes 17.97 s.
1) Model prediction takes 10.35 s. 

Note above running time is based on trainSet_enriched.csv, trainSet_eval.csv and candidateTestSet.txt. Also training process consists of grid search with 2 folds for 3 candidates, totalling 6 fits.


## Weakness & Future work
#### Weakness:
1) Overfitting:
- Considering the majority of categories, model demonstrates a macro-precision of 0.85 with a std of 0.18 during training with enrichment, and 0.58 with a std of 0.2 during evaluation without enrichment.

2) Underfitting:
- There are a small amount of categories, model showing underfitting due to few amount of training data. 

Regarding over- and under-fitting, refer to EDA-metrics notebook for details of analysis. 

3) Model assumption:
 - Naive Bayes model assumes features are independent from each other given a class. However, this may not be fully applicable. For instance, for category 4, 'credit' (222.87) and 'card' (293.66) frequently appear together in a search term. Their sematic relation makes these words not independently and equally contriute to the prediction. This cases also applies for 'debit' (288.55) and 'card'.

 - Naive Bayes model works well for categories with different tokens. But in this task, it is likely that many categories share the same token and high tf-idf scores. 

4) TF-IDF vectorizer;
 - TF-IDF vectorizer calculates the tf-idf score for each token in a term as compared to the whole training data. I think (maybe wrong) there are 2 weakness:
  -- Term frequency: a token's term frequency in a term cannot reflect its importance to a cagegory. Instead, team frequency should be the count of it in a category, i.e., num of appearance of the token in its category.
 -- Inverse document frequency: again this should be on category level, i.e., num of all cats divided by num of cats with this token.

 -- Above logic has been implemented in the codes.

#### Future work:
1) Abbreviation recovery: EDA process finds that for a word such as 'satelite', abbrevation can appear in a form of 'sat'. However, any abbrevation form of a work, such as'sat' will reduce the TD-IDF score of 'satelite'. This may require create a dictionary or use a 3rd party library.

2) High tf-idf score token analysis: frequent tokens analysis among categories helps reveal the similairty of categories. If such, frequent tokens may help group similary categories. 

3) Runtime issue: 
 - Despite of one-off run, data enrichment and lemmatization can be sped up with multi-processing. 
 - TF-IDF vectorizer returns float64 data, making training and prediction process heavily inefficient. Alternative is to convert to float16, though sklearn error is encounted.

4) Data enrichment: 
 - Integrate data enrichment into pipeline
 - Reduce data enrichment amount to 1, and investigate overfitting

5) Data collection:
 - For the categories shows 0 precision both during train and eval, collect more data

6) Data exploration:
 - Noise terms: Some terms are chopped to None after preprocessing. These terms are assumed to noise with no informativeness. Need to review these terms for more actions.
 - Numerical tokens: no preprocessing is done on numerical tokens. But it is worth to analyse TF-IDF scores of numerical tokens to determine their importance before any actions. 
