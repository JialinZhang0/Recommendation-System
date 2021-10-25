from surprise import SVD
from surprise import Dataset, Reader
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# prepare data
csvfile = open('trainset.csv', 'r')
file = open('testset.csv', 'r')

data = []
for line in csvfile :
    data.append(line)

n = 0
for line in file:
    if  n == 0:
        n = 1
    else:
        data.append(line)

datafile = open('data.csv', mode = 'w')
for k in data:
    datafile.write(str(k))
    #datafile.write('\n')
datafile.close()

file_path = os.path.expanduser('data.csv')
#file_path2 = os.path.expanduser('testset.csv')
reader = Reader(line_format = 'user item rating', sep = ',', skip_lines = 1)
#data = Dataset.load_from_file(file_path1, reader = reader)
dataset = Dataset.load_from_file(file_path, reader = reader)

# obtain U, V
algo = SVD(n_factors=100, reg_all=0.05)
data_set = dataset.build_full_trainset()
model = algo.fit(data_set)
#model = evaluate(algo, data, verbose=1, measures=[u'rmse', u'mae'])
item = model.qi #item vector inner
user = model.pu #user vector inner
#print (item, 'next',  np.shape(user))

# item vector transfer
item_raw_iid = []
n = 0
for line in item:
    raw_iid = data_set.to_raw_iid(n)
    item_raw_iid.append(raw_iid)
    n = n+1
#print (item_raw_iid)

csv_file = open('release-year.csv', 'r')
year = []
n = 1
for line in csv_file:
    if n == 1:
        n = 2
    else:
        year.append(line)
#print (year)
reyear = []
for ele in item_raw_iid:
    #print (year[int(ele)])
    reyear.append(int(year[int(ele)]))
#print (reyear)

# user vector transfer
user_raw_uid = []
n = 0
for line in user:
    raw_uid = data_set.to_raw_uid(n)
    user_raw_uid.append(raw_uid)
    n = n+1
#print (user_raw_uid)

genderfile = open('gender.csv', 'r')
gender = []
n = 1
for line in genderfile:
    if n == 1:
        n = 2
    else:
        gender.append(line)

regender = []
for ele in user_raw_uid:
    regender.append(int(gender[int(ele)]))
#print (regender)

#------------------------------------------
# logistic regression on user vector
lr = LogisticRegression(penalty='l2', solver='liblinear')
C = np.linspace(0.001, 10, num=50)
hypeparameters = dict(C=C)
grid = GridSearchCV(lr, hypeparameters, cv=5, verbose=1, return_train_score=True)
best_model_user = grid.fit(user, regender)
#test_accuracy_user = grid.cv_results_['mean_test_score']
#std_test_score_user = grid.cv_results_['std_test_score']
#test_error_user = 1 - test_accuracy_user
#print ('test error of user:', test_error_user)
#print ('test std of user:', std_test_score_user)

C = best_model_user.best_estimator_.get_params()['C']
lamda = 1/C
print ('best lamda:', lamda)
best_score_user = best_model_user.cv_results_['mean_test_score'][best_model_user.best_index_]
cv_error = 1 - best_score_user
best_std_user = best_model_user.cv_results_['std_test_score'][best_model_user.best_index_]
print ('cross validation error', cv_error)
print ('best_std_user', best_std_user)

# random forest on item vector
rfr = RandomForestRegressor(n_jobs=-1, n_estimators=100, oob_score=False)
md = [1, 2, 3, 4, 5]
param_grid = dict(max_depth=md)
grid = GridSearchCV(rfr, param_grid, cv=5, verbose=1, return_train_score=True, refit=True)
best_model_item = grid.fit(item, reyear)
best_score_item = best_model_item.best_score_
print (best_score_item)
para = best_model_item.best_estimator_.get_params()['max_depth']
print ('best max depth', para)
predictyear = best_model_item.predict(item)
print (predictyear)
mse = mean_squared_error(reyear, predictyear)
print ('mse', mse)

#naive model
mean_release_year = np.mean(reyear)
print (mean_release_year)
#print (len(reyear))
mean_year = [mean_release_year]*len(reyear)
mse_naive = mean_squared_error(reyear, mean_year)
print ('mse_naive', mse_naive)