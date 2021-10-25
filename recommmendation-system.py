from surprise import SVD
from surprise import Dataset, Reader, accuracy
#from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from collections import defaultdict
import os

# Load the trainset dataset
file_path1 = os.path.expanduser('trainset.csv')
file_path2 = os.path.expanduser('testset.csv')
reader = Reader(line_format = 'user item rating', sep = ',', skip_lines = 1)
data = Dataset.load_from_file(file_path1, reader = reader)
testdata = Dataset.load_from_file(file_path2, reader = reader)

# tune hyperparameter
param_grid = {'n_factors': [50, 75, 100, 125, 150], 'reg_all': [0.01, 0.05, 0.1, 0.5, 1] }
grid_search = GridSearchCV(SVD, param_grid, cv=5, joblib_verbose=2, n_jobs=-1, return_train_measures=True, measures=[u'rmse', u'mae'], refit=True)
grid_search.fit(data)
print (grid_search.best_params['mae'])


#evaluate recommendation 1
test1 = testdata.build_full_trainset()
testing1 = test1.build_testset()
#print (testing1)
predictions1 = grid_search.test(testing1, verbose=False)
#print (predictions1)
#grid_search.test(testdata)
test1_accuracy = accuracy.mae(predictions1, verbose=True)
print ('mean absolute error', test1_accuracy)

#evaluate recommendation 2
train = data.build_full_trainset()
training = train.build_anti_testset(fill=True)
predictions2 = grid_search.test(training, verbose=False)
def get_top_n(predictions2, n=5):
    # first map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions2:
        top_n[uid].append((iid,est))
    # the n sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

#the n predict ratings for all pairs (u, i) that are not in the training set.
top_n = get_top_n(predictions2, n=5)

#print the recommended items for each user
for uid, user_ratings in top_n.items():
    #print (uid, [iid for(iid,_) in user_ratings])
    for c, value in enumerate(top_n[uid]):
        #print (uid)
        for uid_1, iid_1, true_r_1, est_1, _ in predictions1:
            if value[0] == iid_1 and uid == uid_1:
                #print ('iid1', iid_1)
                #print ('true', true_r_1)
                top_n[uid][c] = (iid_1, true_r_1)
                break
            else:
                top_n[uid][c] = (value[0], 2.0)
#print (top_n)

total = 0
for uid in top_n:
    for i in top_n[uid]:
        total += i[1]
    #total += top_n[uid][1]
#print (total)
#print (len(top_n))
mean = total/(len(top_n)*5)
print ('overall average rating', mean)


