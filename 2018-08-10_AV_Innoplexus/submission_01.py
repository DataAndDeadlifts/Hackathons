import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk import wordpunct_tokenize
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

vectorizer = TfidfVectorizer(input='content', analyzer='word')
svd = TruncatedSVD(n_components=500, n_iter=5, random_state=27)

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#After we use get_text, use nltk's clean_html function.
def nltkPipe(soup_text):
    #Convert to tokens
    tokens = [x.lower() for x in wordpunct_tokenize(soup_text)]
    text = nltk.Text(tokens)
    #Get lowercase words. No single letters, and no stop words
    words = [w.lower() for w in text if w.isalpha() and len(w) > 1 and w.lower() not in stop_words]
    #Remove prefix/suffixes to cut down on vocab
    stemmer = EnglishStemmer()
    words_nostems = [stemmer.stem(w) for w in words]
    return words_nostems

def getTitleTokens(html):
    soup = BeautifulSoup(html,'html.parser')
    soup_title = soup.title
    if soup_title != None:
        soup_title_text = soup.title.get_text()
        text_arr = nltkPipe(soup_title_text)
        return text_arr
    else:
        return []
    
def getBodyTokens(html):
    soup = BeautifulSoup(html,'html.parser')
    #Get the text body
    soup_para = soup.find_all('p')
    soup_para_clean = ' '.join([x.get_text() for x in soup_para if x.span==None and x.a==None])
    text_arr = nltkPipe(soup_para_clean)
    return text_arr

#Build the model
def get_html(in_df):
    keep_cols = ["Webpage_id","Tag"]
    use_df = in_df[keep_cols]
    html_reader_obj = pd.read_csv(data_dir+'html_data.csv',iterator=True, chunksize=10000)
    frames = []
    match_indices = use_df['Webpage_id'].values.tolist()
    print(len(match_indices),' indices left...')
    while len(match_indices) > 0:
        for chunk in html_reader_obj:
            merge_df = pd.merge(use_df,chunk,how='inner',on='Webpage_id')
            merge_indices = merge_df['Webpage_id'].values.tolist()
            match_indices = [x for x in match_indices if x not in merge_indices]
            print(len(match_indices),' indices left...')
            frames.append(merge_df)
    #Process HTMl for bags of words of the body and title.
    process_df = pd.concat(frames)
    print("Getting tokens...")
    title_tokens = process_df['Html'].progress_apply(getTitleTokens)
    body_tokens = process_df['Html'].progress_apply(getBodyTokens)
    process_df['all_tokens'] = title_tokens + body_tokens
    process_df.drop(['Html'],axis=1,inplace=True)
    print("Done!")
    return process_df

def build_model():
    """Return the estimator and the object to transform the test data."""
    print("Getting HTML tokens")
    data_dir = "../data/2018-08-10_AV_Innoplexus/"

    html_data = pd.read_csv(data_dir+'html_data.csv',iterator=True, chunksize=1000)
    
    train_df = pd.read_csv(data_dir+'train.csv')
    
    #Get tokens
    train_df_tokens = get_html(train_df)
    #Fit_transform to tdfif matrix
    train_df_tdif = vectorizer.fit_transform(train_df_tokens['all_tokens'])
    #Prune unneeded features
    svd_array = svd.fit_transform(train_df_tdif)
    
    vector_features = vectorizer.get_feature_names()
    eigen_features = [vector_features[i] for i in svd.components_[0].argsort()[::-1]][:500]

    train_df_svd = pd.DataFrame(svd_array,columns=eigen_features)
    train_df_svd['Tag'] = train_df['Tag']
    
    tags = train_df_svd['Tag'].unique().tolist()
    tags.sort()

    tag_dict = {key: value for (key, value) in zip(tags,range(len(tags)))}

    train_df_svd['Tag_encoded'] = train_df_svd['Tag'].map(tag_dict)
    train_df_svd = svd_df.drop('Tag',axis=1)
    
    exported_pipeline = make_pipeline(
        StackingEstimator(
            estimator=ExtraTreesClassifier(
                bootstrap=False, criterion="gini", max_features=0.2, 
                min_samples_leaf=11, min_samples_split=17, n_estimators=100)
        ),
        ExtraTreesClassifier(
            bootstrap=False, criterion="entropy", max_features=0.5, 
            min_samples_leaf=6, min_samples_split=9, n_estimators=100
        )
    )
    
    x_cols = [x for x in train_df_svd.columns if x != "Tag_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(
        train_df_svd[x_cols],
        train_df_svd['Tag_encoded'],
        test_size=0.33
    )
    
    exported_pipeline.fit(X_train, y_train)
    return exported_pipeline, vectorizer, svd, tag_dict

def prep_test(vectorizer_obj, svd_obj):
    """Transform test dataset for predicting."""
    test_df = pd.read_csv(data_dir+'test.csv')
    #Get the HTMl
    test_df_tokens = get_html(test_df)
    #Transform to tdfif matrix
    test_df_tdif = vectorizer_obj.transform(test_df_tokens['all_tokens'])
    #Prune unneeded features
    test_svd_array = svd_obj.transform(test_df_tdif)
    
    vector_features = vectorizer_obj.get_feature_names()
    eigen_features = [vector_features[i] for i in svd_obj.components_[0].argsort()[::-1]][:500]
    #Map to dataframe
    test_df_svd = pd.DataFrame(test_svd_array,columns=eigen_features)
    test_df_svd['Tag'] = test_df['Tag']
    return test_df_svd

def main():
    #Get the model
    model, vectorizer_obj, svd_obj, tag_dict = build_model()
    #Prep the test set
    test_df = prep_test(vectorizer_obj, svd_obj)
    predictions = model.predict(test_df)
    return predictions