# Fake and Real News Classifier

A text classifier that reads a news article and guesses whether it comes from the "real" or the "fake" side of a well-known Kaggle dataset. I built it with Python, pandas and scikit-learn while doing a one-month data science and ML training program, plus self-study on the side.

The model itself is simple (TF-IDF and logistic regression). Most of my time went into the data, because the first version scored 98.5% and I wanted to know why.

## Dataset

[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) on Kaggle (the ISOT dataset). It has two files, `True.csv` (21,417 articles) and `Fake.csv` (23,481 articles), each with `title`, `text`, `subject` and `date`.

The CSV files are not in this repo because of their size. Download them from Kaggle and put them next to the notebook.

## What I did

1. Loaded both files, added a `label` column (0 = real, 1 = fake), stacked them and shuffled.
2. Explored the data and found a few problems (see below).
3. Cleaned the text: lowercase, remove links, keep letters only, remove stopwords with NLTK.
4. Split 80/20 with `stratify` and `random_state=42`.
5. Turned the text into numbers with TF-IDF (5,000 words, `min_df=5`), fitting it on the training set only.
6. Trained a logistic regression model and checked it with a confusion matrix and precision/recall.
7. Looked at the words the model weighted most, removed the ones that were clearly source markers, and retrained to see how much accuracy depended on them.

## Cleaning log

| Step | Rows removed | Rows left |
|---|---|---|
| Start (both files combined) | | 44,898 |
| Duplicate `text` | 6,252 (6,026 fake, 226 real) | 38,646 |
| `text` shorter than 100 characters | 305 (all fake) | 38,341 |
| Fewer than 10 words (link-only posts) | 3 (all fake) | 38,338 |

After cleaning there are 21,191 real articles (55.3%) and 17,147 fake ones (44.7%). A model that always answered "real" would get about 55%, so that is the number to beat.

## Things I found in the data

These matter more than the final score.

**The `subject` column gives the answer away.** Six subjects (News, politics, left-news, Government News, US_News, Middle-east) appear only in fake articles, and the other two (politicsNews, worldnews) appear only in real ones. A model could reach 100% from this column alone, so I never used it as a feature.

**Almost every real article contains "Reuters".** It appears in 21,154 of the 21,191 real articles (99.8%) and in 215 of the 17,147 fake ones (1.3%). Real articles usually start with something like "WASHINGTON (Reuters) -", so a model can learn to spot the source and not the writing.

**Empty and tiny texts were nearly all fake.** Before removing duplicates, 630 of the 631 blank texts were fake. Many of the short ones were captions for embedded videos or tweets. I removed them so the model could not use "no text" as a clue.

**Duplicates were mostly fake too.** About a quarter of the fake articles repeated an earlier text, against roughly 1% of the real ones. Leaving them in would put copies of the same article in both the training and test sets.

**Length is a weak signal.** The median article is 359 words for real and 380 for fake. The mean gap is bigger (385 vs 433) because a few very long articles pull it up: 230 fake articles are over 1,500 words, against 34 real ones.

## Results

| Version | Words removed from the text | Test accuracy |
|---|---|---|
| First model | none | 98.5% |
| Round A | reuters, washington | 97.8% |
| Round B | plus weekdays, "nov", "edt", "via", "getty", "image", "featured" and a few more | 97.0% |
| Final model | plus city names, "images", "video", "us", "said" and a few more (32 words in total) | 96.3% |

The first model made 115 mistakes on 7,668 test articles: 81 fake articles called real and 34 real articles called fake. Precision and recall were about 0.98 to 0.99 for both classes.

The accuracy dropped slowly as I removed words, which tells me the model is not leaning on one or two shortcuts. Real and fake articles in this dataset come from different websites, and each site has its own vocabulary and formatting, so the model can pick up many small clues. I could remove the obvious ones, but not all of them.

## Limitations

- The model recognises the writing style and source patterns of these two groups of websites. It does not check whether a claim is true. A false story written like a wire report would probably be labelled real.
- Almost all of the data is US politics from 2015 to 2018, so it may do badly on other topics or other years.
- The training articles are all at least 100 characters and 10 words long, so very short inputs are unreliable.
- Removing words by hand is not a clean fix. A better approach would be to strip the dateline from the raw text before cleaning, and to test on articles from a completely different source.
- The score comes from a single train/test split. I did not use cross-validation.

## Files

| File | What it is |
|---|---|
| `Fake_and_Real_News_Detector.ipynb` | The full notebook, from loading the data to the final model |
| `model.joblib` | The trained logistic regression model |
| `vectorizer.joblib` | The TF-IDF vectorizer the model was trained with |
| `leak_words.joblib` | The list of words removed before vectorizing |

The model only works with its own vectorizer and the same word list, so the three files go together.

## Running it

```
pip install pandas scikit-learn nltk matplotlib seaborn joblib
```

Put `True.csv` and `Fake.csv` in the same folder as the notebook and run the cells from top to bottom. To use the saved model on a new article:

```python
import joblib

model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")
leak_words = set(joblib.load("leak_words.joblib"))

cleaned = clean_text(article)          # the clean_text function from the notebook
cleaned = " ".join(w for w in cleaned.split() if w not in leak_words)
print(model.predict_proba(vectorizer.transform([cleaned])))   # [P(real), P(fake)]
```

## What I would do next

- Remove the Reuters-style dateline from the raw text and compare.
- Try Naive Bayes and a linear SVM on the same split.
- Test on articles from sources that are not in this dataset.
- Add cross-validation so the accuracy numbers are less dependent on one split.
