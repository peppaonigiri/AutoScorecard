from pyecharts.charts import WordCloud
from pyecharts import options as opts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# 1.词云图
def preprocess_plot_wordcloud(word, weight, title):
    tuple_lst = []
    for i in range(len(word)):
        tuple_lst.append((word[i], weight[i]))

    word_cloud = WordCloud()
    word_cloud.add(series_name=title, data_pair=tuple_lst, word_size_range=[30, 80], width=800, height=600)
    # word_cloud.load_javascript()
    # word_cloud.render_notebook()


# 2. 词袋模型 bag of features, BOF
def preprocess_BOF(corpus, num):
    vec = CountVectorizer(ngram_range=(1, num))
    X_ngram = vec.fit_transform(corpus)
    feature_names = vec.get_feature_names()
    X_ngram_array = X_ngram.toarray()
    return feature_names, X_ngram_array

#词袋模型
# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer()
# corpus = [
#     'This is a very good class',
#     'students are very very very good',
#     'This is the third sentence',
#     'Is this the last doc',
#     'PS teacher Mei is very very handsome'
# ]
# X = vectorizer.fit_transform(corpus)
# vectorizer.get_feature_names()
# X.toarray()
# vec = CountVectorizer(ngram_range=(1,3))
# X_ngram = vec.fit_transform(corpus)
# vec.get_feature_names()
# X_ngram.toarray()


# 3. TF-IDF
def preprocess_TF_IDF(corpus):
    tfidf_vec = TfidfVectorizer()
    tfidf_X = tfidf_vec.fit_transform(corpus)
    feature_names = tfidf_vec.get_feature_names()
    tfidf_X_array = tfidf_X.toarray()
    return feature_names, tfidf_X_array