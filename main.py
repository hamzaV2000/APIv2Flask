import datetime
import json

import pandas as pd
import numpy as np
import warnings
import re
from pickle import dump, load

from flask_cors import CORS
from gevent.pywsgi import WSGIServer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, Response
from flask_restful import Api, Resource
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from pymysql import connect

warnings.filterwarnings('ignore')

stopwords = set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                 "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
                 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
                 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
                 'after',
                 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                 'further',
                 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                 'more',
                 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o',
                 're',
                 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
                 "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
                 "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
                 "weren't",
                 'won', "won't", 'wouldn', "wouldn't"])

df_books_processed = pd.read_csv('result/books_with_authors_names.csv').dropna()

database = connect(host="192.168.100.12"
                   , user="testing", password="Test1234@", database="heaven", port=3306)
# df_books_users = pd.read_csv('result/betaReviews2.csv')

df_books_users = pd.read_sql("select * from reviews", database)
last = datetime.datetime.now()


def update_df_books_users():
    global last
    global database
    global df_books_users
    myCursor = database.cursor()
    sql = "select * from reviews where timestamp > %s"
    myCursor.execute(sql, last)
    last = datetime.datetime.now()
    field_names = [i[0] for i in myCursor.description]
    df = pd.DataFrame(myCursor.fetchall(), columns=field_names)
    df = df.set_index('id')
    df_books_users = df_books_users.reindex(df_books_users.index.union(df.index))
    df_books_users.update(df)
    database.commit()


class TopN(Resource):
    def get(self):
        return Response(top_book_average_rating().to_json(orient="records"), mimetype='application/json')


class RecommendBySimilarUsers(Resource):
    def get(self, user_id):
        print(df_books_users.shape)
        return Response(similar_user_df(user_id).to_json(orient="records"), mimetype='application/json')


class RecommendBySimilarBooks(Resource):
    def get(self, title):
        return Response(top_50_similar_title_books(title).to_json(orient="records"), mimetype='application/json')


class UserFavorites(Resource):
    def get(self, user_id):
        return Response(user_favorites(user_id).to_json(orient="records"), mimetype='application/json')


class Search(Resource):
    def get(self, domain, query):
        if domain == 'title':
            return Response(top_50_similar_title_books(query).to_json(orient="records"), mimetype='application/json')
        elif domain == 'all':
            result = top_50_with_similar_genres(query).sort_values(by='book_average_rating', ascending=False).head(5)
            result1 = books_by_author(query).sort_values(by='book_average_rating', ascending=False).head(5)
            result2 = top_50_similar_title_books(query).sort_values(by='book_average_rating', ascending=False).head(5)
            result3 = top_50_similar_description_books(query).sort_values(by='book_average_rating',
                                                                          ascending=False).head(5)
            result = pd.concat([result, result1, result2, result3], axis=0)
            return Response(result.to_json(orient="records"),
                            mimetype='application/json')
        elif domain == 'description':
            return Response(top_50_similar_description_books(query).to_json(orient="records"),
                            mimetype='application/json')
        elif domain == 'genre':
            return Response(top_50_with_similar_genres(query).to_json(orient="records"),
                            mimetype='application/json')
        elif domain == 'author':
            return Response(books_by_author(query).to_json(orient="records"), mimetype='application/json')
        else:
            return "not found"


"""""

vectorizer_description = load(open("content/vectorizer_description.pkl", "rb"))
tfidf_description = load(open("content/tfidf_description.pkl", "rb"))
vectorizer_title_fc = load(open("content/vectorizer_title.pkl", "rb"))
tfidf_title_fc = load(open("content/tfidf_title.pkl", "rb"))
vectorizer_genres_fc = load(open("content/vectorizer_genres.pkl", "rb"))
tfidf_genres_fc = load(open("content/tfidf_genres.pkl", "rb"))
"""""
vectorizer_description = TfidfVectorizer(stop_words=stopwords)

tfidf_description = vectorizer_description.fit_transform(df_books_processed['book_description'])

with open('content/vectorizer_description.pkl', 'wb') as p:
    dump(vectorizer_description, p)

with open('content/tfidf_description.pkl', 'wb') as p:
    dump(tfidf_description, p)

with open("content/vectorizer_description.pkl", "rb") as p:
    vectorizer_description = load(p)

with open("content/tfidf_description.pkl", "rb") as p:
    tfidf_description = load(p)

vectorizer_title_fc = TfidfVectorizer(stop_words=stopwords)
tfidf_title_fc = vectorizer_title_fc.fit_transform(df_books_processed['mod_title'])

with open('content/vectorizer_title.pkl', 'wb') as p:
    dump(vectorizer_title_fc, p)

with open('content/tfidf_title.pkl', 'wb') as p:
    dump(tfidf_title_fc, p)

with open("content/vectorizer_title.pkl", "rb") as p:
    vectorizer_title_fc = load(p)

with open("content/tfidf_title.pkl", "rb") as p:
    tfidf_title_fc = load(p)

vectorizer_genres_fc = TfidfVectorizer(stop_words=stopwords)
tfidf_genres_fc = vectorizer_genres_fc.fit_transform(df_books_processed['genres'])

with open('content/vectorizer_genres.pkl', 'wb') as p:
    dump(vectorizer_genres_fc, p)

with open('content/tfidf_genres.pkl', 'wb') as p:
    dump(tfidf_genres_fc, p)

with open("content/vectorizer_genres.pkl", "rb") as p:
    vectorizer_genres_fc = load(p)

with open("content/tfidf_genres.pkl", "rb") as p:
    tfidf_genres_fc = load(p)


def top_book_average_rating():
    top50_highest_rated_books = df_books_processed[
        (df_books_processed['book_average_rating'] >= 4.50) & (df_books_processed['ratings_count'] > 3000.0)]
    return top50_highest_rated_books.sort_values(by='book_average_rating', ascending=False)


def books_by_author(author):
    author = author.lower()
    df_books_processed['name2'] = df_books_processed['name'].str.lower()
    books_byauthor = df_books_processed[df_books_processed['name2'].str.contains(author)]
    return books_byauthor.sort_values(by='book_average_rating', ascending=False).head(50)


def top_concised_books():
    top_50_concised_books = df_books_processed[
        (df_books_processed['num_pages'] <= 300) & (df_books_processed['ratings_count'] > 3000.0) & (
                df_books_processed['book_average_rating'] >= 4.50)].sort_values(by='book_average_rating',
                                                                                ascending=False)
    return top_50_concised_books.sort_values(by='book_average_rating', ascending=False)


def top_50_similar_title_books(title):
    vectorizer = vectorizer_title_fc
    query = title
    print(title)
    processed = re.sub("[^a-zA-Z0-9 ]", "", query.lower())
    print(processed)
    query_vec = vectorizer.transform([processed])
    similarity = cosine_similarity(query_vec, tfidf_title_fc).flatten()
    indices = np.argpartition(similarity, -50)[-50:]
    results = df_books_processed.iloc[indices]
    return results


def top_50_with_similar_genres(query):
    print(query)
    vectorizer = vectorizer_genres_fc
    # genres = df_books_processed[df_books_processed['book_id'] == int(query)]['genres']
    # print(genres.values)
    processed = re.sub("[^a-zA-Z0-9]", " ", query.lower())
    print(processed)
    query_vec = vectorizer.transform([processed])
    similarity = cosine_similarity(query_vec, tfidf_genres_fc).flatten()
    indices = np.argpartition(similarity, -50)[-50:]
    results = df_books_processed.iloc[indices]
    return results


def top_50_similar_description_books(query):
    vectorizer = vectorizer_description
    processed = re.sub("[^a-zA-Z0-9 ]", "", query.lower())
    query_vec = vectorizer.transform([processed])
    similarity = cosine_similarity(query_vec, tfidf_description).flatten()
    indices = np.argpartition(similarity, -50)[-50:]
    results = df_books_processed.iloc[indices]
    return results


def similar_user_df(user_id):
    update_df_books_users()
    df_liked_books = df_books_users[df_books_users['user_id'] == user_id]
    liked_books = set(df_liked_books['book_id'])
    top_5_liked_books = df_liked_books.sort_values(by='user_rating', ascending=False)['book_id'][:5]
    similar_user = \
        df_books_users[(df_books_users['book_id'].isin(top_5_liked_books)) & (df_books_users['user_rating'] > 4)][
            'user_id']
    data = df_books_users[(df_books_users['user_id'].isin(similar_user))].merge(df_books_processed, on='book_id')
    return popular_recommendation(data, liked_books)


def popular_recommendation(recs, liked_books):
    all_recs = recs["book_id"].value_counts()
    all_recs = all_recs.to_frame().reset_index()
    all_recs.columns = ["book_id", "book_count"]
    all_recs = all_recs.merge(recs, how="inner", on="book_id")
    all_recs["score"] = all_recs["book_count"] * (all_recs["book_count"] / all_recs["ratings_count"])
    popular_recs = all_recs.sort_values("score", ascending=False)
    popular_recs_unbiased = popular_recs[~popular_recs["book_id"].isin(liked_books)].drop_duplicates(
        subset=['title_without_series'])
    return popular_recs_unbiased.sort_values(by='book_average_rating', ascending=False).head(12)


def user_favorites(user_id):
    update_df_books_users()
    result = df_books_users[df_books_users['user_id'] == user_id]
    result = result[result['user_rating'] > 4.0]
    result = df_books_processed.merge(result, on='book_id')
    return result


app = Flask(__name__)
api = Api(app)
CORS(app)

api.add_resource(TopN, "/topn")
api.add_resource(UserFavorites, "/userFavorites/<int:user_id>")
api.add_resource(RecommendBySimilarUsers, "/recommendBySimilarUsers/<int:user_id>")
api.add_resource(RecommendBySimilarBooks, "/recommendBySimilarBooks/<string:title>")
api.add_resource(Search, "/search/<string:domain>/<string:query>")

if __name__ == "__main__":
    print('serving')
    http_server = WSGIServer(('0.0.0.0', 5001), app)
    http_server.serve_forever()
