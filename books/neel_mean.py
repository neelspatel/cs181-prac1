import numpy as np
import util

# This makes predictions based on the mean rating for each user in the
# training data.  When there are no training data for a user, it
# defaults to the global mean.

pred_filename  = 'pred-user-mean.csv'
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

# Turn the list of users into a dictionary.
# Store data for each user to keep track of the per-user average.
cities = {}
states = {}
nations = {}


users = {}
for user in user_list:
    users[user['user']] = { 'total': 0, # For storing the total of ratings.
                            'count': 0, # For storing the number of ratings.
                            'location': user["location"],
                            'age': user["age"],
                            }

    cns = util.citystatenation(user['location'])
    if cns:        
        cities[cns["city"]] = { 'total': 0, 'count': 0}
        states[cns["state"]] = { 'total': 0, 'count': 0}
        nations[cns["nation"]] = { 'total': 0, 'count': 0}

# Turn the list of books into an ISBN-keyed dictionary.
# Store data for each book to keep track of the per-book average.
books = {}
for book in book_list:
    books[book['isbn']] = { 'total': 0, # For storing the total of ratings.
                            'count': 0, # For storing the number of ratings.
                            }

# Iterate over the training data to compute means.
for rating in training_data:
    user_id = rating['user']
    users[user_id]['total'] += rating['rating']
    users[user_id]['count'] += 1


    books[rating['isbn']]['total'] += rating['rating']
    books[rating['isbn']]['count'] += 1

    cns = util.citystatenation(users[user_id]['location'])
    if cns:
        cities[cns["city"]]['total'] += rating['rating']
        cities[cns["city"]]['count'] += 1
        states[cns["state"]]['total'] += rating['rating']
        states[cns["state"]]['count'] += 1
        nations[cns["nation"]]['total'] += rating['rating']
        nations[cns["nation"]]['count'] += 1

no_user = 0
no_book = 0

# Make predictions for each test query.
for query in test_queries:

    user = users[query['user']]
    book = books[query['isbn']]    

    if user['count'] == 0 :
        # Perhaps we did not having any ratings in the training set.
        # In this case, make a global mean prediction.
        query['rating'] = mean_rating
        no_user = no_user + 1

    elif book['count'] == 0:
        # Perhaps we did not having any ratings in the training set.
        # In this case, make a global mean prediction.
        query['rating'] = mean_rating
        no_book = no_book + 1

    else:
        # Predict the average for this user.

        query['rating'] = (10 * (float(user['total']) / user['count']) + 0.5*(float(book['total']) / book['count']) )/10.5

        #benchmark: user mean (around 0.73)
        #query['rating'] = (float(user['total']) / user['count'])
        
        #benchmark: book mean (around 0.96)
        #query['rating'] = (float(book['total']) / book['count'])

# Write the prediction file.
util.write_predictions(test_queries, pred_filename)

print "No book: " + str(no_book)
print "No user: " + str(no_user)
print "Other: " + str(len(test_queries) - no_user - no_book)

#print util.rmse(test_queries)
