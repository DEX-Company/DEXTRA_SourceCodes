### This code is used to create some popular benchmark algorithms for the MINDEF challenge
import numpy as np
import pandas as pd
import pylab as P
import DataProcessing as dp



##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@##
def split_rows(df, cln_list):
	'''
	In this dataset, from certain column onwards, all the data is missing. This is demonstrated as figure below:
	-------------------------
	|           |       |    |
	|           |       |    |
	--------------------------
	|           |       |
	|           |       |
	---------------------
	|           |
	|           |
	-------------
	In the 1st part, the rows have all the data
	In the 2nd part, those rows miss some columns
	In the 3rd part, those rows miss even more columns
	'''
	header = df.columns.tolist()
	cln_idx_1 = header.index(cln_list[1])
	cln_idx_2 = header.index(cln_list[2])
	df_1 = df.ix[df[cln_list[0]].notnull(), 0:cln_idx_1]
	df_2 = df.ix[df[cln_list[1]].notnull(), 0:cln_idx_2]
	df_3 = df.ix[df[cln_list[2]].notnull(), 0::]
#!	print df_3
	return [df_1, df_2, df_3]

def s2(df_1, df_2):
	'''
	This method takes in two dataframes, train and test, as input, and then standarsize some columns according to their properties
	'''
	from sklearn.preprocessing import StandardScaler

	if (df_1.shape[1] == df_2.shape[1]):
		print "Two Data Frames have the same # of columns! Good to go!!!"
		print
	num_of_rows_1 = df_1.shape[0]
	num_of_rows_2 = df_2.shape[0]
	df = pd.concat([df_1, df_2])
	header_numerical = df.columns[df.dtypes.map(lambda x: x!='object')].tolist()

	print df[header_numerical].info()
	df[header_numerical] = StandardScaler().fit_transform(df[header_numerical])
	print df[header_numerical].describe()
	return df[0: num_of_rows_1], df[num_of_rows_1: num_of_rows_1 + num_of_rows_2]

def impute(df_1, df_2):
	'''
	This method takes in two dataframes, train and test, as input, and then impute some columns of numerical values according to their properties
	'''
	from sklearn.preprocessing import StandardScaler

	if (df_1.shape[1] == df_2.shape[1]):
		print "Two Data Frames have the same # of columns! Good to go!!!"
		print
	num_of_rows_1 = df_1.shape[0]
	num_of_rows_2 = df_2.shape[0]
	df = pd.concat([df_1, df_2])
	header_numerical = df.columns[df.dtypes.map(lambda x: x!='object')].tolist()

	for cln in header_numerical:
		df[cln] = df[cln].fillna(df[cln].mode().ix[0]) # Fill NA with mode, the most frequent values
#	print df[header_numerical].head(5)

	return df[0: num_of_rows_1], df[num_of_rows_1: num_of_rows_1 + num_of_rows_2]

def one_hot_encoder(df_1, df_2):
	'''
	This method takes train and test dataframes as input, and then convert the categorical data into values.
	The principal is similar to the one hot encoder in sklearn
	'''
	if (df_1.shape[1] == df_2.shape[1]):
		print "Two Data Frames have the same # of columns! Good to go!!!"
		print
	num_of_rows_1 = df_1.shape[0]
	num_of_rows_2 = df_2.shape[0]
	df = pd.concat([df_1, df_2])
	header_prefix = df.columns[df.dtypes.map(lambda x: x=='object')].tolist()
	df = pd.get_dummies(df, prefix = header_prefix, prefix_sep = '_')
	return df[0: num_of_rows_1].values, df[num_of_rows_1: num_of_rows_1 + num_of_rows_2].values, df.columns.tolist()

def PCA_analysis(train, test, Num):
	'''
	This methos takes two inputs, training data (without label) and testing data; it computes the principal component, and then return training
	and testing data
	'''
	from sklearn.decomposition import PCA
	num_of_rows_1 = train.shape[0]
	num_of_rows_2 = test.shape[0]
	data = np.vstack((train, test))

	pca = PCA(n_components=int(data.shape[1] * Num), copy=True, whiten=False)
	data = pca.fit_transform(data)
	print data.shape
	return data[:num_of_rows_1, :], data[num_of_rows_1: num_of_rows_1 + num_of_rows_2, :]

def LogLoss_cap(x):
	cap_high = 0.9999
	cap_low = 1.0 - cap_high
	for i in xrange(len(x)):
		x[i] = max(min(cap_high, x[i]), cap_low)
	return x

def features_select(train_data, train_label, test_data):
	'''
	This method is for feature selection using l1
	'''
	from sklearn.linear_model import LogisticRegression as lr

	model = lr(penalty='l1')
	model.fit(train_data, train_label)
	idx_true_false = abs(model.coef_) > 0.00001 # Select the index of features with non-zero coefficients
	idx_true_false = idx_true_false.flatten()
	return train_data[:, idx_true_false], test_data[:, idx_true_false], idx_true_false

def algo_predict_LR(train_data, train_label, test_data, test_id):
	from sklearn.linear_model import LogisticRegression as lr

	model = lr()
	model.fit(train_data, train_label)
	y = model.predict_proba(test_data)[:, 1]
#	y = LogLoss_cap(y) # This cap function turns out to be useless in Logistic regression
#	print model.classes_
	return dict(zip(test_id.astype(str), y))

def algo_predict_LR_II(train_data, train_label, test_data, test_id):
	'''
	This method is mainly for feature selection using l1 and model testing, such as testing various methods in the algorithm
	'''
	from sklearn.linear_model import LogisticRegression as lr
	log_loss = dp.mindef().LogLoss_II

	model = lr(penalty='l2')
	model.fit(train_data, train_label)
	y = model.predict_proba(test_data)[:, 1]
#	y = LogLoss_cap(y) # This cap function turns out to be useless in Logistic regression
#	print model.classes_
	print model.coef_
	print abs(model.coef_) > 0.00001 # Select the index of features with non-zero coefficients
	print train_data.shape
	print model.coef_.shape
	print "Log Loss error of training data is : ", log_loss(train_label, model.predict_proba(train_data)[:, 1])
	return dict(zip(test_id.astype(str), y))

def algo_predict_LR_III(train_data, train_label, test_data, test_id):
	'''
	This method uses logistic regression with cross validation implemented
	'''
	from sklearn.linear_model import LogisticRegressionCV as lr
	from sklearn.metrics import log_loss

#	model = lr(Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10], cv = 5, refit = False, scoring = 'log_loss', n_jobs = -1, verbose = 3)
#	model = lr(Cs = 9, cv = 12, refit = False, scoring = 'log_loss', n_jobs = -1, verbose = 3) # using cross validation
	model = lr(Cs = 9, refit = False, scoring = 'log_loss', n_jobs = -1, verbose = 3) # No cross validation
	# Set regularization para Cs from 10^-4 to 10^4, K-fold cross validation cv = 5, scoring function as log loss,
	# use all cores to compute [n_jobs=-1], and output some details [verbose = 3]
	model.fit(train_data, train_label)
	y = model.predict_proba(test_data)[:, 1]
	print "Parameters in this model", model.get_params()
	print "Scores in the cross validation: ", model.scores_
#	print "Inverse of regularization strength is: ", model.Cs_
#	y = LogLoss_cap(y) # This cap function turns out to be useless in Logistic regression
#	print model.classes_
	return dict(zip(test_id.astype(str), y))

def algo_predict_tree(train_data, train_label, test_data, test_id, header):
	from sklearn import tree
	from sklearn.externals.six import StringIO
	import pydot
	from os import system
	clf = tree.DecisionTreeClassifier(criterion='entropy')
	clf = clf.fit(train_data, train_label)

	'''
	dot_data = StringIO()
	tree.export_graphviz(clf, out_file=dot_data)
	graph = pydot.graph_from_dot_data(dot_data.getvalue())
	graph.write_png("./MINDEF_Tree.png")
	'''
	dotfile = open("./dtree2.dot", 'w')
	tree.export_graphviz(clf, out_file = dotfile, feature_names=header, max_depth=3)
	dotfile.close()
	system("dot -Teps dtree2.dot -o ./dtree2.eps")

def algo_predict_SVM(train_data, train_label, test_data, test_id):
#	http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
	from sklearn.svm import SVC as svc

	model = svc(kernel='linear', probability=True)
	model.fit(train_data, train_label)
	y = model.predict_proba(test_data)[:, 1]
#	y = LogLoss_cap(y) # This cap function turns out to be useless in Logistic regression
#	print model.classes_
	return dict(zip(test_id.astype(str), y))

def algo_predict_SVM2(train_data, train_label, test_data, test_id):
#	http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html
	from sklearn.svm import NuSVC as svc

	model = svc(nu=0.1, kernel='linear', probability=True)
	model.fit(train_data, train_label)
	y = model.predict_proba(test_data)[:, 1]
#	y = LogLoss_cap(y) # This cap function turns out to be useless in Logistic regression
#	print model.classes_
	return dict(zip(test_id.astype(str), y))

def combine_results(results):
	'''
	The input results contain 3 prediction results from 3 different sets of features. The method combines these inputs with different weights.
	'''
	weight1 = [0.4, 0.6]
	weight2 = [0.2, 0.2, 0.6]
	results_combined = results[0]
	'''
	for per_id in results[0].keys():
		try:
			tmp = results[1][per_id]
			try:
				tmp = results[2][per_id]
				results_combined[per_id] = results[0][per_id] * weight2[0] + results[1][per_id] * weight2[1] + results[2][per_id] * weight2[2]
			except KeyError:
				results_combined[per_id] = results[0][per_id] * weight1[0] + results[1][per_id] * weight1[1]
		except KeyError:
			pass
	'''
	return np.array(zip(results_combined.keys(), results_combined.values()))

def split_transform(df_1, df_2, cln_list):
	'''
	Split the training/test data into 3 parts based on the observation that some people donnot have certain columns
	Splitting accords to columns
	And then transform columns of catergorized values into binary vectors, aka one-hot-encoder
	-------------------------
	|           |       |    |
	|           |       |    |
	--------------------------
	|           |       |
	|           |       |
	---------------------
	|           |
	|           |
	-------------
	'''

	df_1_new =0
	df_2_new = 1
	return df_1_new, df_2_new

def data_munging():
	'''
	Clean the data, impute missing values
	'''
	'''
## Step 0. Read the training data into data frame format using pandas, and explore the data using various pandas methods
	train_path = "/Volumes/Data/Dextra/Mindef/Data20150726/ForParticipant/HR_Retention_2013_training.csv"
	train_df = pd.read_csv(train_path, header = 0)
	train_header = train_df.columns.tolist()
#!	print train_header
#!	print train_df
#!	print train_df.head(3)
#!	print train_df.tail(3)
#!	print type(train_df)
#!	print train_df.dtypes
#!	print train_df.info()
#!	print train_df.describe()
#!	print train_df['AGE'][0:10]
#!	print train_df.AGE[0:10]
#!	print type(train_df.AGE)
#!	print train_df['AGE'].mean()
#!	print train_df['AGE'].median()
#!	print train_df[['PERID', 'RESIGNED', 'AGE']]
#!	print train_df[train_df['AGE'] > 30]
#!	print train_df[train_df['AGE'] > 30]['AGE']
#!	print train_df[train_df['VOC'].isnull()]['AGE'].count()
#!	print train_df.head(1)
#!	train_df['AGE'].hist()
#!	P.show()
#!	print train_df['RESIGN_DATE']
	'''
## Step 1. Read the training/test data into data frame, and remove some columns
	mindef = dp.mindef_benchmark()
	train_path = "/Volumes/Data/Dextra/Mindef/Data20150726/ForParticipant/HR_Retention_2013_training.csv"
	test_path = "/Volumes/Data/Dextra/Mindef/Data20150726/ForParticipant/HR_Retention_2013_to_be_predicted.csv"

#	train_cln_remove = ['PERID', 'RESIGN_DATE', 'RESIGNATION_MTH', 'RESIGNATION_QTR', 'RESIGNATION_YEAR', 'AGE_GROUPING', 'STATUS', 'EMPLOYEE_GROUP']
#	test_cln_remove = ['AGE_GROUPING', 'EMPLOYEE_GROUP']

	train_cln_remove = ['PERID', 'RESIGN_DATE', 'RESIGNATION_MTH', 'RESIGNATION_QTR', 'RESIGNATION_YEAR', 'AGE_GROUPING', 'STATUS']
	test_cln_remove = ['AGE_GROUPING']

	train_df = mindef.read_remove_clns(train_path, train_cln_remove)
#!	print train_df[0:4].head(3)
#!	print

	test_df = mindef.read_remove_clns(test_path, test_cln_remove)
#!	print test_df[0:4].head(3)

	'''
## Step 2. Split the training/test data into 3 parts based on the observation that some people don't have certain columns
##         And then transform columns of catergorized values into binary vectors, aka one-hot-encoder
	cln_list = ['GENDER', 'VOC', 'UPGRADED_LAST_3_YRS']
	train_df, test_df = split_transform(train_df, test_df, cln_list)

	'''
## Step 2. Split the training/test data into 3 sectors based on the observation that some people don't have certain columns
	cln_list = ['GENDER', 'VOC', 'UPGRADED_LAST_3_YRS']
	train_3_datasets = split_rows(train_df, cln_list)

	test_3_datasets = split_rows(test_df, cln_list)
## Step 3. Check whether the new data frames got null cells or not
	'''
	for df in train_3_datasets:
		print df.info()
		print
	for df in test_3_datasets:
		print df.info()
		print
	'''
	# From the check, we find that almost no null values in those data frames any more.
	# There are at most 1 or 2 rows in those data frames, which means the data is quite clean.
	# However, we still need to pay special attention to those null values, especially those in columns of numerical values.
	# We need to impute those missing values, which will be addressed in the following
## Step 4. For each of these 3 splitted data frames in training/test, convert the columns of categorical values into
##         binary vectors, aka one-hot-encoder: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html
	results = [{}, {}, {}] # results is a list of 3 dictionaries, and each dictionary is the result for each of 3 sectors
	for i in xrange(1):
	## Step 4.1 Further strip the datasets, separate the label in training data, and id in test data
		train_label = train_3_datasets[i]['RESIGNED'].values
		test_id = test_3_datasets[i]['PERID'].values

		train_3_datasets[i] = train_3_datasets[i].drop('RESIGNED', axis = 1)
		test_3_datasets[i] = test_3_datasets[i].drop('PERID', axis = 1)
	## Step 4.2 Impute the missing data in those columns with numerical values
		train_3_datasets[i], test_3_datasets[i] = impute(train_3_datasets[i], test_3_datasets[i])
	## Step 4.3 Standardize the datasets
#		train_3_datasets[i], test_3_datasets[i] = s2(train_3_datasets[i], test_3_datasets[i])
	## Step 4.4 Convert categorical values into numerical values
		train_data, test_data, header = one_hot_encoder(train_3_datasets[i], test_3_datasets[i])
	## Step 4.5 Principle Component Analysis
#		train_data, test_data = PCA_analysis(train_data, test_data, Num = 0.4)
#		--> It is proven not useful here.
	## Step 4.6 Select features using l1 regularization
		train_data, test_data, idx_true_false = features_select(train_data, train_label, test_data)
		print "Remained features are: ", np.array(header)[idx_true_false]
		print idx_true_false
#		print train_data
#		print test_data
#		print train_label
#		print test_id
#		results[i] = algo_predict_LR_III(train_data, train_label, test_data, test_id)
		algo_predict_tree(train_data, train_label, test_data, test_id, list(np.array(header)[idx_true_false]))
#		results[i] = algo_predict_SVM(train_data, train_label, test_data, test_id)
#		results[i] = algo_predict_SVM2(train_data, train_label, test_data, test_id)
#	results = combine_results(results)
#	mindef.array_to_csv(np.array(results), './LogisticRegression.csv', header = np.array(['PERID', 'RESIGNED']))
##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@##















if __name__ == "__main__":
## Step 1. Clean the data, impute missing values
	data_munging()
## Step 2. Run different classication algorithms on the data