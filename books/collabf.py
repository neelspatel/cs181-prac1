import numpy as np
import util
import cPickle as pickle
import sys
import operator

# This makes predictions based on the mean rating for each user in the
# training data.  When there are no training data for a user, it
# defaults to the global mean.

def dist(a,b):
    return np.linalg.norm(a-b)

def nearest_pred(cur_user, users, prefs, index_to_get, k):
    rated = []

    #create the list of pref vectors
    for user in users:
        rated.append(prefs[user])
    
    #loops through and find the closest pref vector, recording its index
    min_index = 0
    min_distance = float("inf")
    results = []
    for i in range(len(rated)):
        cur_rating = rated[i]
        cur_dist = dist(cur_user, cur_rating)
        if cur_dist < min_distance:
            min_distance = cur_dist
            min_index = i
        results.append({"distance": cur_dist, "rating": cur_rating})

    sorted_results = sorted(results, key=operator.itemgetter('distance')) 

    closest = sorted_results[:k]
    total = 0.0
    length = len(closest)
    count = 0.0

    for current in closest:
        total += current["rating"][index_to_get] * (length - count)
        count += 1.0

    #result = total/((2 * k - count) * (count + 1) / 2)
    result = total / ((count) * (count + 1) / 2)

    if result < 0 or result > 5:
        print "Error: " + str(result)
    
    return result
    
pred_filename  = 'pred-cf-user-mean.csv'
#train_filename = 'half-ratings-train.csv'
#test_filename  = 'half-ratings-test.csv'
train_filename = 'ratings-train.csv'
test_filename  = 'ratings-test.csv'
user_filename  = 'users.csv'
book_filename  = 'books.csv'

training_data  = util.load_train(train_filename)
test_queries   = util.load_test(test_filename)
user_list      = util.load_users(user_filename)
book_list      = util.load_books(book_filename)

# Compute the global mean rating for a fallback.
num_train = len(training_data)
mean_rating = float(sum(map(lambda x: x['rating'], training_data)))/num_train
print "The global mean rating is %0.3f." % (mean_rating)

#USER MEAN
users = {}
for user in user_list:
    users[user['user']] = { 'total': 0, # For storing the total of ratings.
                            'count': 0, # For storing the number of ratings.
                            }
    
# Iterate over the training data to compute means.
for rating in training_data:
    user_id = rating['user']
    users[user_id]['total'] += rating['rating']
    users[user_id]['count'] += 1


#get list of books
books_indices = {}
books_users = {}
current_index = 0

for book in book_list:
    isbn = book["isbn"]
    if not isbn in books_indices:
        books_indices[isbn] = current_index
        current_index += 1

#for each user, creates vector of prefs
prefs = {}
num_books = len(books_indices)

print "We have " + str(num_books) + " books"

count = 0
for rating in training_data:
    if count % 1000 == 0:
        print str(count)
    count += 1
    book = rating["isbn"]
    user = rating["user"]
    if book in books_indices:
        if user in prefs:
            index = books_indices[book]
            prefs[user][index] = rating['rating']
        else:
            vector = np.zeros(num_books)
            prefs[user] = vector

        if not book in books_users:
            books_users[book] = []

        if not user in books_users[book]:
            books_users[book].append(user)
    else:
        print "Book not in here: " + str(book)

count = 0
book_except_count = 0
user_except_count = 0
# Make predictions for each test query.
for query in test_queries:
    if count % 1000 == 0:
        print str(count)

    cur_user = query['user']
    cur_book = query['isbn']
    cur_book_index = books_indices[cur_book]

    #print "About to get pref"

    if cur_user in prefs:
        cur_pref = prefs[cur_user]

        #print "About to get cur book"
        
        if cur_book in books_users:  
            rated_users = books_users[cur_book]

            #print "About to make prediction"
            prediction = nearest_pred(cur_user, rated_users, prefs, cur_book_index, 15)
            
            if users[cur_user]['count'] == 0:
                # Perhaps we did not having any ratings in the training set.
                # In this case, make a global mean prediction.
                query['rating'] = prediction

            else:
                # Predict the average for this user.
                query['rating'] = ((float(users[cur_user]['total']) / users[cur_user]['count']) + prediction) / 2
            
        else:
            #print "Book not rated... fuck"
            
            if users[cur_user]['count'] == 0:
                # Perhaps we did not having any ratings in the training set.
                # In this case, make a global mean prediction.
                query['rating'] = mean_rating

            else:
                # Predict the average for this user.
                query['rating'] = float(users[cur_user]['total']) / users[cur_user]['count']

            book_except_count += 1
    else:
        #print "User not rated... fuck"
        query['rating'] = mean_rating
        user_except_count += 1

    count += 1

print "Book Excepted " + str(book_except_count)
print "User Excepted " + str(user_except_count)

# Write the prediction file.
util.write_predictions(test_queries, pred_filename)


#print util.rmse(test_queries[:1000])
