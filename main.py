import datetime
import json

import pandas as pd
import numpy as np
import warnings
import re
from pickle import dump, load

import pymysql
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
from scipy.sparse import coo_matrix
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, Response
from flask_restful import Api, Resource
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from pymysql import connect
from gevent import monkey

monkey.patch_all()
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

host = "192.168.100.12"
user = "testing"
password = "Test1234@"
database = connect(host=host
                   , user=user, password=password, database="heaven", port=3306)
# df_books_users = pd.read_csv('result/betaReviews2.csv')

df_books_users = pd.read_sql("select * from reviews", database)
df_books_users.set_index('id', inplace=True)

df_books_ratings = pd.read_sql("select id, book_id, rating_count, average_rating from ratings", database)

df_books_processed = pd.read_csv('result/books_with_authors_names.csv')[
    ['book_id', 'mod_title', 'book_description', 'genres', 'name', 'num_pages']].dropna()

last = datetime.datetime.now()
le9 = preprocessing.LabelEncoder()
le9.fit(df_books_users['user_id'])
df_books_users['user_id_mapped'] = le9.transform(df_books_users['user_id'])

le1 = preprocessing.LabelEncoder()
le1.fit(df_books_users['book_id'])
df_books_users['book_id_mapped'] = le1.transform(df_books_users['book_id'])

df_books_users_ratings = df_books_users.merge(df_books_ratings, on='book_id')


def update_books_ratings(changed_book_ids):
    global last
    global df_books_ratings
    global df_books_users
    global df_books_users_ratings
    database1 = connect(host=host
                        , user=user, password=password, database="heaven", port=3306)
    for value in changed_book_ids:
        sql = "select id,book_id, rating_count, average_rating from ratings where book_id = %s"
        myCursor = database1.cursor()
        myCursor.execute(sql, value)
        field_names = [i[0] for i in myCursor.description]
        df = pd.DataFrame(myCursor.fetchall(), columns=field_names)
        df = df.set_index('id')
        df_books_ratings = df_books_ratings.reindex(df_books_ratings.index.union(df.index))
        df_books_ratings.update(df)
    database1.commit()
    database1.close()
    myCursor.close()
    le9.fit(df_books_users['user_id'])
    df_books_users['user_id_mapped'] = le9.transform(df_books_users['user_id'])
    le1.fit(df_books_users['book_id'])
    df_books_users['book_id_mapped'] = le1.transform(df_books_users['book_id'])
    df_books_users_ratings = df_books_users.merge(df_books_ratings, on='book_id')


def update_df_books_users():
    global last
    global df_books_users
    database1 = connect(host=host
                        , user=user, password=password, database="heaven", port=3306)

    sql = "select * from reviews_changes"
    myCursor = database1.cursor()
    myCursor.execute(sql)
    myCursor.close()
    field_names = [i[0] for i in myCursor.description]
    changes_df = pd.DataFrame(myCursor.fetchall(), columns=field_names)
    changes_df = changes_df.set_index('id')
    if changes_df.size <= 0:
        # print("changes is 0")
        database1.commit()
        database1.close()
        return

    myCursor = database1.cursor()
    changedBook_ids = set()
    for user_id in changes_df['user_id'].values:
        sql = "select * from reviews where user_id = %s"
        # print("select * from reviews where user_id =" + str(user_id))
        myCursor.execute(sql, user_id)
        last = datetime.datetime.now()
        field_names = [i[0] for i in myCursor.description]
        df = pd.DataFrame(myCursor.fetchall(), columns=field_names)
        # print("df", df.head(5))
        changedBook_ids.update(set(df['book_id']))
        df = df.set_index('id')
        # if user_id in df_books_users['user_id'].values:
        #     df_books_users.update(df)
        # else:
        #    df_books_users = pd.concat([df_books_users, df], axis=0)

        # df_books_users.update(df)

        df_books_users = (pd.concat([df_books_users, df], ignore_index=True, sort=False)
                          .drop_duplicates(['user_id', 'book_id'], keep='last'))
        sql = "delete from reviews_changes where user_id = %s"
        myCursor.execute(sql, user_id)

    database1.commit()
    database1.close()
    myCursor.close()
    update_books_ratings(changedBook_ids)


class TopN(Resource):
    def get(self):
        return Response(top_book_average_rating().to_json(orient="records")
                        , mimetype='application/json')


class RecommendBySimilarUsers(Resource):
    def get(self, user_id):
        return Response(similar_users(user_id).to_json(orient="records")
                        , mimetype='application/json')


class RecommendBySimilarTitle(Resource):
    def get(self, title):
        return Response(top_50_similar_title_books(title).to_json(orient="records")
                        , mimetype='application/json')


class RecommendBySimilarBook(Resource):
    def get(self, book_id):
        return Response(similar_item_recommendation(book_id).to_json(orient="records")
                        , mimetype='application/json')


class UserFavorites(Resource):
    def get(self, user_id):
        return Response(user_favorites(user_id).to_json(orient="records")
                        , mimetype='application/json')


class Search(Resource):
    def get(self, domain, query):
        if domain == 'title':
            return Response(top_50_similar_title_books(query).to_json(orient="records")
                            , mimetype='application/json')
        elif domain == 'all':
            result = top_50_with_similar_genres(query).head(25)
            result1 = books_by_author(query).head(25)
            result2 = top_50_similar_title_books(query).head(25)
            result3 = top_50_similar_description_books(query).head(25)
            result = pd.concat([result2, result3, result1, result], axis=0)
            return Response(result.to_json(orient="records"),
                            mimetype='application/json')
        elif domain == 'description':
            return Response(top_50_similar_description_books(query).to_json(orient="records"),
                            mimetype='application/json')
        elif domain == 'genre':
            return Response(top_50_with_similar_genres(query).to_json(orient="records"),
                            mimetype='application/json')
        elif domain == 'author':
            return Response(books_by_author(query).to_json(orient="records")
                            , mimetype='application/json')
        else:
            return "not found"


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
"""""


def top_book_average_rating():
    update_df_books_users()
    top50_highest_rated_books = df_books_ratings[
        (df_books_ratings['average_rating'] >= 4.2) & (df_books_ratings['rating_count'] >= 20.0)]
    return top50_highest_rated_books.sort_values(by='average_rating', ascending=False)['book_id'].head(100)


def books_by_author(author):
    author = author.lower()
    df_books_processed['name2'] = df_books_processed['name'].str.lower()
    books_byauthor = df_books_processed[df_books_processed['name2'].str.contains(author)]
    return books_byauthor.merge(df_books_processed, on='book_id').merge(df_books_ratings, on='book_id').sort_values(
        by='average_rating', ascending=False).head(50)['book_id']


def top_concised_books():
    top_50_concised_books = df_books_processed[
        (df_books_processed['num_pages'] <= 300) & (df_books_processed['rating_count'] > 3000.0) & (
                df_books_processed['average_rating'] >= 4.50)].sort_values(by='average_rating',
                                                                           ascending=False)
    return top_50_concised_books.merge(df_books_processed, on='book_id').sort_values(
        by='average_rating', ascending=False).head(50)['book_id']


def top_50_similar_title_books(title):
    vectorizer = vectorizer_title_fc
    query = title
    print(title)
    processed = re.sub("[^a-zA-Z0-9 ]", "", query.lower())
    query_vec = vectorizer.transform([processed])
    similarity = cosine_similarity(query_vec, tfidf_title_fc).flatten()
    indices = np.argpartition(similarity, -50)[-50:]
    results = df_books_processed.iloc[indices]
    return results['book_id']


def top_50_with_similar_genres(query):
    print(query)
    vectorizer = vectorizer_genres_fc
    # genres = df_books_processed[df_books_processed['book_id'] == int(query)]['genres']
    # print(genres.values)
    processed = re.sub("[^a-zA-Z0-9]", " ", query.lower())
    query_vec = vectorizer.transform([processed])
    similarity = cosine_similarity(query_vec, tfidf_genres_fc).flatten()
    indices = np.argpartition(similarity, -50)[-50:]
    results = df_books_processed.iloc[indices]
    return results['book_id']


def top_50_similar_description_books(query):
    vectorizer = vectorizer_description
    processed = re.sub("[^a-zA-Z0-9 ]", "", query.lower())
    query_vec = vectorizer.transform([processed])
    similarity = cosine_similarity(query_vec, tfidf_description).flatten()
    indices = np.argpartition(similarity, -50)[-50:]
    results = df_books_processed.iloc[indices]
    return results['book_id']


def similar_user_df(user_id):
    update_df_books_users()
    df_liked_books = df_books_users[df_books_users['user_id'] == user_id]
    liked_books = set(df_liked_books['book_id'])
    top_5_liked_books = df_liked_books.sort_values(by='user_rating', ascending=False)['book_id'][:5]
    similar_user = \
        df_books_users[(df_books_users['book_id'].isin(top_5_liked_books)) & (df_books_users['user_rating'] >= 4)][
            'user_id']
    data = df_books_users[(df_books_users['user_id'].isin(similar_user))].merge(
        df_books_processed, on='book_id').merge(df_books_ratings, on="book_id")
    return popular_recommendation(data, liked_books)


def popular_recommendation(recs, liked_books):
    all_recs = recs["book_id"].value_counts()
    all_recs = all_recs.to_frame().reset_index()
    all_recs.columns = ["book_id", "book_count"]
    all_recs = all_recs.merge(recs, how="inner", on="book_id")
    all_recs["score"] = all_recs["book_count"] * (all_recs["book_count"] / all_recs["rating_count"])
    popular_recs = all_recs.sort_values("score", ascending=False)
    popular_recs_unbiased = popular_recs[~popular_recs["book_id"].isin(liked_books)].drop_duplicates(
        subset=['mod_title'])
    return popular_recs_unbiased.head(50)


def similar_users(user_id):
    update_df_books_users()
    books_liked_by_user = set(df_books_users[df_books_users['user_id'] == user_id]['book_id'])
    count_other_similar_users = df_books_users[df_books_users['book_id'].isin(books_liked_by_user)][
        'user_id'].value_counts()
    df_similar_user = count_other_similar_users.to_frame().reset_index()
    df_similar_user.columns = ['user_id', 'matching_book_count']
    top_onepercent_similar_users = df_similar_user[
        df_similar_user['matching_book_count'] >= np.percentile(df_similar_user['matching_book_count'], 99)]
    top_users = set(top_onepercent_similar_users['user_id'])
    df_similar_user = df_books_users[(df_books_users['user_id'].isin(top_users))][
        ['user_id_mapped', 'book_id_mapped', 'user_rating', 'user_id', 'book_id']]
    ratings_mat_coo = coo_matrix(
        (df_similar_user["user_rating"], (df_similar_user["user_id_mapped"], df_similar_user["book_id_mapped"])))
    ratings_mat = ratings_mat_coo.tocsr()
    my_index = list(le9.transform([user_id]))[0]
    similarity = cosine_similarity(ratings_mat[my_index, :], ratings_mat).flatten()
    similar_users_index = np.argsort(similarity)[-1:-51:-1]
    book_title_liked_by_user = set(
        df_books_users[df_books_users['book_id'].isin(books_liked_by_user)].sort_values(
            by='user_rating', ascending=False))

    df_similar_users = df_similar_user[(df_similar_user["user_id_mapped"].isin(similar_users_index)) & (
        ~df_similar_user['book_id'].isin(books_liked_by_user))].merge(
        df_books_processed, on='book_id').merge(df_books_ratings, on="book_id")
    print(df_similar_users.head(5))
    return popular_recommendation(df_similar_users, book_title_liked_by_user)


def recommendation(df_similar_users_refined):
    all_recs = df_similar_users_refined['book_id'].value_counts()
    all_recs = all_recs.to_frame().reset_index()
    all_recs.columns = ["book_id", "book_count"]
    all_recs_book_id = list(all_recs['book_id'])
    all_recs_new = df_books_processed[df_books_processed['book_id'].isin(all_recs_book_id)]
    all_recs_new = all_recs_new.merge(all_recs, on='book_id', how='inner')
    all_recs_new['score'] = all_recs_new['book_count'] * (all_recs_new['book_count'] / all_recs_new['rating_count'])
    return all_recs_new.sort_values(by='score', ascending=False)[['book_id']].head(50)


def user_favorites(user_id):
    update_df_books_users()
    result = df_books_users[df_books_users['user_id'] == user_id]
    print(result.head(5))
    result = result[result['user_rating'] >= 4.0]
    result = df_books_processed.merge(result, on='book_id')
    return result['book_id']


def similar_item_recommendation(book_id):
    global df_books_users_ratings
    users_who_liked_book = set(df_books_users_ratings[df_books_users_ratings['book_id'] == book_id]['user_id'])
    books_id_remaining = df_books_users_ratings[
        (df_books_users_ratings['user_id'].isin(list(users_who_liked_book)))]
    ratings_mat_coo = coo_matrix((books_id_remaining["user_rating"],
                                  (books_id_remaining["book_id_mapped"], books_id_remaining["user_id_mapped"])))
    ratings_mat = ratings_mat_coo.tocsr()
    my_index = list(le1.transform([book_id]))[0]
    similarity = cosine_similarity(ratings_mat[my_index, :], ratings_mat).flatten()
    similar_books_index = np.argsort(similarity)[-1:-51:-1]
    score = [(score, book) for score, book in enumerate(similar_books_index)]
    df_score = pd.DataFrame(score, columns=['score', 'book_id_mapped'])
    df_similar_books_to_recommend = (
        df_books_users_ratings[(df_books_users_ratings['book_id_mapped'].isin(list(similar_books_index)))].merge(
            df_score, on='book_id_mapped'))[['book_id', 'score']]
    unique_df_similar_books_to_recommend = df_similar_books_to_recommend.drop_duplicates(keep='first')
    final_books = (df_books_processed[df_books_processed['book_id'].isin(
        set(unique_df_similar_books_to_recommend['book_id'].values))].merge(unique_df_similar_books_to_recommend,
                                                                            on='book_id')).sort_values(by='score')
    return final_books['book_id']


app = Flask(__name__)
api = Api(app)
CORS(app)

api.add_resource(TopN, "/python/topn")
api.add_resource(UserFavorites, "/python/userFavorites/<int:user_id>")
api.add_resource(RecommendBySimilarUsers, "/python/recommendBySimilarUsers/<int:user_id>")
api.add_resource(RecommendBySimilarTitle, "/python/recommendBySimilarTitle/<string:title>")
api.add_resource(RecommendBySimilarBook, "/python/recommendBySimilarBook/<int:book_id>")
api.add_resource(Search, "/python/search/<string:domain>/<string:query>")

if __name__ == "__main__":
    print('serving')
    http_server = WSGIServer(('0.0.0.0', 5001), app)
    http_server.serve_forever()
