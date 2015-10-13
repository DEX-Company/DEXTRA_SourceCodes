#! This is a module that contains methods for data processing, such as write to csv, read csv into array and etc
import csv as csv
import numpy as np
import pandas as pd
import time
from sklearn.cross_validation import StratifiedShuffleSplit as sss
import math
from datetime import datetime as dt


class read_write(object):
	'''Contains many methods for read/write data from/to csv files. Now seldomly use because built-in method in pandas is more convenient.
	However, some methods are still useful as it can be used for creating data files for participants in DEXTRA
	'''
	def csv_to_array(self, file_path_name, mode = 'rb'):
		file_object = open(file_path_name, mode)
		csv_file_oject = csv.reader(file_object)
		header = np.array([csv_file_oject.next()])
		csv_file_list = []
		for rows in csv_file_oject:
			csv_file_list.append(rows)
		csv_file_array = np.array(csv_file_list)
		return (header, csv_file_array)

	def csv_s_to_array(self, file_path_names, mode = 'rb'):
		file_path_name = file_path_names[0]
		file_object = open(file_path_name, mode)
		csv_file_oject = csv.reader(file_object)
		array = np.array([csv_file_oject.next()])
		file_object.close()
		for path in file_path_names:
			(header, arr) = self.csv_to_array(path, mode)
			array = np.vstack((array, arr))
		return (header, array[1:, :])

	def csv_to_2_arrays(self, file_path_name, mode = 'rb'):
		file_object = open(file_path_name, mode)
		csv_file_oject = csv.reader(file_object)
		header = np.array([csv_file_oject.next()])
		csv_file_list = []
		for rows in csv_file_oject:
			csv_file_list.append(rows)
		csv_file_array = np.array(csv_file_list)
		row_cut_I = np.where(csv_file_array[:, 0] == "-1")[0][0]
		array_I = csv_file_array[:row_cut_I, :]
		row_cut_II = np.where(csv_file_array[:, 0] == "-2")[0][0]
		array_II = csv_file_array[row_cut_I + 1:row_cut_II, :]
		return (header, array_I, array_II)

	def tsv_to_array(self, file_path_name, mode = 'rb'):
		file_object = open(file_path_name, mode)
		tsv_file_oject = csv.reader(file_object, delimiter='\t')
		header = np.array([tsv_file_oject.next()])
		tsv_file_list = []
		for rows in tsv_file_oject:
			tsv_file_list.append(rows)
		tsv_file_array = np.array(tsv_file_list)
		return (header, tsv_file_array)

	def array_to_csv(self, array, file_path, header = 0):
		open_file_object = open(file_path, 'wb')
		write_file_object = csv.writer(open_file_object)
		if (header != 0):
			write_file_object.writerow(list(header.flatten()))
	#	print
	#	print header
	#	print list(header)
		for row in array:
			write_file_object.writerow(list(row))
		open_file_object.close()

	def two_arrays_to_csv(self, array_I, array_II, file_path, header = None):
		open_file_object = open(file_path, 'wb')
		write_file_object = csv.writer(open_file_object)
		write_file_object.writerow(list(header.flatten()))
		boundary = np.array([["-1", "DEXTRA"]])
		boundary_II = np.array([["-2", "DEXTRA"]])
		array = np.vstack((array_I, boundary, array_II, boundary_II))
		for row in array:
			write_file_object.writerow(list(row))
		open_file_object.close()

	def array_to_tsv(self, array, file_path, header = None):
		open_file_object = open(file_path, 'wb')
		write_file_object = csv.writer(open_file_object, delimiter = "\t")
		write_file_object.writerow(list(header.flatten()))
		for row in array:
			write_file_object.writerow(list(row))
		open_file_object.close()

class create_files_for_participant(object):
	def sample_submission(self, header, data, cln_name):
		num_rows = np.size(data[:,0]) # number of rows
		cln_num = np.where(header == cln_name)[1][0]
		id_column = data[:,cln_num].reshape((num_rows,1)) # extract the id column of the dataset
		random_assignment = np.random.uniform(0.0, 1.0, size = (num_rows,1)).astype(str) # -> This is to generate uniform float value between 0 and 1
	#!	print random_assignment
	#!	print file_paths
		return np.hstack((id_column, random_assignment))

class data_clean(object):
	'''
	This class include methods for data cleaning. The main packages used here are pandas and numpy
	'''
	def read_remove_clns(self, path, cln_list):
		df = pd.read_csv(path, header = 0) # Read the csv file into data frame
		return df.drop(cln_list, axis = 1) # Remove columns in the list cln_list

	def read_remove_rows(self, path_I, path_II, cln):
		'''This function reads 2 csv files into data frames df_1 and df_2, and for the common column 'cln', remove rows
		in df_1 with entry appearing in df_2

		:return -> new data frame of df_1 with rows that are in df_2 removed
		'''
		with open(path_I) as f1, open(path_II) as f2:
			df1 = pd.read_csv(path_I, header = 0)
			df2 = pd.read_csv(path_II, header = 0)

			arr_1 = df1[cln].values.astype(str)
			arr_2 = df2[cln].values.astype(str)

			arr_1 = np.char.lower(arr_1)
			arr_2 = np.char.lower(arr_2)

			idx_true_false = np.invert(np.in1d(arr_1, arr_2))
			df_new = df1[idx_true_false]
		return df_new

class datetimeparse(object):
	'''This class contains the method to parse time string in dataframe, and separate time into year month day weekday time and etc.
	'''
	def __init__():
		pass

	@classmethod
	def split_beta(cls, df, time_str_format = "%m/%d/%Y %H:%M", columns = ['month', 'day'], special_columns = None):
		'''This method is the prototype, very slow. Please check the new version <split2> below
		'''
		num_clns = len(columns)
		num_rows = df.size
		cln_name = df.name
		new_cln_names = map(lambda x: cln_name + '_' + x, columns)
		new_df = pd.DataFrame(np.zeros([num_rows, num_clns]), columns = new_cln_names)
		dict_time_retrive = {'year':0, 'month':1, 'day':2, 'hour':3, 'minute':4, 'second':5, 'weekday':6, 'day_in_year':7}
		# datetime obj turple contains: tm_year=2014, tm_mon=8, tm_mday=31, tm_hour=22, tm_min=15, tm_sec=0, tm_wday=6, tm_yday=243, tm_isdst=-1
#!		print new_cln_names
#!		print new_df.info()
		# The for loop takes 3-5 mins, too long, need improvement, such as using apply in data frame
		if special_columns:
			num_clns_2 = 1
			new_cln_names_2 = ['total_seconds']
			new_df_2 = pd.DataFrame(np.zeros([num_rows, num_clns_2]), columns = new_cln_names_2)
			for i, date_time in enumerate(df):
				date_time_obj = dt.strptime(df[i], time_str_format)
				date_time_tuple = date_time_obj.timetuple()
				for cln, cln2 in zip(new_cln_names, columns):
					new_df[cln][i] = date_time_tuple[dict_time_retrive[cln2]]

				new_df_2[i] = (date_time_obj - special_columns).total_seconds()
			new_df = pd.concat([new_df, new_df_2], axis = 1)
		else:
			for i, date_time in enumerate(df):
				date_time_obj = dt.strptime(df[i], time_str_format)
				date_time_tuple = date_time_obj.timetuple()
				for cln, cln2 in zip(new_cln_names, columns):
					new_df[cln][i] = date_time_tuple[dict_time_retrive[cln2]]

		return new_df

	@classmethod
	def split(cls, df, time_str_format = "%m/%d/%Y %H:%M", columns = ['month', 'day'], special_columns = None):
		'''This classmethod is an upgraded one of <split>, the  main change is that the for for loop is replaced with DataFrame.apply
		After test, it is proved to be much faster than the prototype
		'''
		num_clns = len(columns)
		num_rows = df.size
		cln_name = df.name
		new_cln_names = map(lambda x: cln_name + '_' + x, columns)
		new_df = pd.DataFrame(np.zeros([num_rows, num_clns]), columns = new_cln_names)
		dict_time_retrive = {'year':0, 'month':1, 'day':2, 'hour':3, 'minute':4, 'second':5, 'weekday':6, 'day_in_year':7}
		# datetime obj turple contains: tm_year=2014, tm_mon=8, tm_mday=31, tm_hour=22, tm_min=15, tm_sec=0, tm_wday=6, tm_yday=243, tm_isdst=-1
#!		print new_cln_names
#!		print new_df.info()
		# The for loop takes 3-5 mins, too long, need improvement, such as using apply in data frame
		df = df.apply(lambda x: dt.strptime(x, time_str_format))
		df_tuple = df.apply(lambda x: x.timetuple())
		if special_columns:
			num_clns_2 = 1
			for cln, cln2 in zip(new_cln_names, columns):
				new_df[cln] = df_tuple.apply(lambda x: x[dict_time_retrive[cln2]])

			new_df_2 = df.apply(lambda x: (x - special_columns).total_seconds())
			new_df_2.name = cln_name + '_' + 'total_seconds'
			new_df = pd.concat([new_df, new_df_2], axis = 1)
		else:
			for cln, cln2 in zip(new_cln_names, columns):
				new_df[cln] = df_tuple.apply(lambda x: x[dict_time_retrive[cln2]])

		return new_df

class evaluation_metric(object):
	def LogarithmicLoss(self, submission, public, private):
		'''
		This method take 3 inputs, the submission array, the backend array for public score, and backend array for private score
		'''
		dict_submission = dict(zip(submission[:, 0], submission[:, 1]))
		dict_public = dict(zip(public[:, 0], public[:, 1]))
		dict_private = dict(zip(private[:, 0], private[:, 1]))

		sum_public = 0.0
		for key in dict_public.keys():
			truth = int(dict_public[key])
			if (truth == 1):
				prediction = max(float(dict_submission[key]), pow(10, -15))
			else:
				prediction = max(1.0 - float(dict_submission[key]), pow(10, -15))
			sum_public = sum_public + math.log10(prediction)

		sum_private = 0.0
		for key in dict_private.keys():
			truth = int(dict_private[key])
			if (truth == 1):
				prediction = max(float(dict_submission[key]), pow(10, -15))
			else:
				prediction = max(1.0 - float(dict_submission[key]), pow(10, -15))
			sum_private = sum_private + math.log10(prediction)

		return - sum_public/len(dict_public.keys()), - sum_private/len(dict_private.keys())

	def LogLoss(self, truth, prediction):
		'''
		This method take 2 inputs: the array of true value and the array of predicted value; and the the method will evaluate the prediction against
		truth using Log Loss metric; each array has two columns
		'''
		dict_truth = dict(zip(truth[:, 0], truth[:, 1]))
		dict_predicted = dict(zip(prediction[:, 0], prediction[:, 1]))

		score = 0.0
		for key in dict_truth.keys():
			truth = int(dict_truth[key])
			if (truth == 1):
				predicted = min(1, max(float(dict_predicted[key]), pow(10, -15)))
			else:
				predicted = min(1, max(1.0 - float(dict_predicted[key]), pow(10, -15)))
			score = score + math.log10(predicted)
		return - score/len(dict_truth.keys())

	def LogLoss_II(self, truth, prediction):
		'''
		This method take 2 inputs: the array of true value and the array of predicted value; and the the method will evaluate the prediction against
		truth using Log Loss metric; each array has only one column
		'''
		N = len(truth)

		score = 0.0
		for i in xrange(N):
			truth_value = int(truth[i])
			if (truth_value == 1):
				predicted = min(1, max(float(prediction[i]), pow(10, -15)))
			else:
				predicted = min(1, max(1.0 - float(prediction[i]), pow(10, -15)))
			score = score + math.log10(predicted)
		return - score/N

	@classmethod
	def Precision(cls, truth, prediction):
		'''This metric is used for calculate precision of binary classification, mainly for implementing on DEXTRA
		'''
		pass

	@classmethod
	def Accuracy(cls, truth, prediction):
		pass

	@classmethod
	def F1(cls, truth, prediction):
		pass

	@classmethod
	def AUC(cls, truth, prediction):
		pass

class engineer_feature(object):
	'''Contains various methods used for feature engineering.
	'''
	def __init__(self, df_train, df_test = None):#, df, param = {'pd_describe': True, 'zero': False}):
		'''Initialise parameters for feature engineering, mainly determine which feature to be engineered
		'''
		frames = [df_train, df_test]
		self.df = pd.concat(frames, keys=['x', 'y'])
		self.method_list = []
		self.df_final = None
#		_key_lst = ['pd_describe', 'zero']
#		for key in key

	def pd_feature(self):
		'''This method will add features to the data frame using the statistics of pandas built-in method
		lessons learned from pandas apply method: I. apply usually assume the data type is object, which is
		often not desirable. DO CONVERT it to what you want, otherwise you get wrong result. 
		II. Series is quite differ from dataframe, at least in terms of shape. <apply> assumes
		the passed/returned data is series of shape (n,), NOT dataframe of shape (n,1)!!!
		'''
		self.df_pd = self.df.apply(lambda x: x.astype('float64').describe(), axis=1)
		self.method_list.append(self.df_pd)
		return self.df_pd.ix['x'], self.df_pd.ix['y']

	def _get_corr(self, df_series, y):
		'''Calculate the covariance between feature/column and label

		Parameters
		----------
		df_series : pandas.series, column of training data
		y : 1-D numpy array, label of training data
		'''
		x = df_series.values
		idx = np.isnan(x)
		idx = np.invert(idx)
#		print idx
		x = x[idx]
#		print x
		y = y[idx]

		n = len(x) + 0.0
		mean1 = sum(x) / n
		mean2 = sum(y) / n

		variance1 = 0
		variance2 = 0

		covariance = 0
		for i in range(int(n)):
			a = x[i] - mean1
			b = y[i] - mean2
			covariance += a*b
			variance1 += a*a
			variance2 += b*b
		correlation = covariance/math.sqrt(variance1*variance2)

		return correlation

	def corr_feature(self, y):
		'''Evaluate the covariance of each feature/column with respect to label

		Parameters
		----------
		y : 1-D numpy array, label of training data
		'''
		df_train = self.df.ix['x']
		df_corr = df_train.apply(self._get_corr, axis=0, args=(y,))
		print df_corr

	def aggregate_features(self):
		'''Concatenates all the features created here, and return the 
		'''
		self.df_final = self.df
		for df in self.method_list:
			self.df_final = pd.concat([self.df_final, df], axis = 1)
		return self.df_final.ix['x'], self.df_final.ix['y']

	def to_csv(self, path_train='./train_data_with_new_features.csv', path_test='./test_data_with_new_features.csv', train_index=None, train_label=None, test_index=None, index=False):
		if self.df_final is not None:
			df_train = pd.concat([train_index, self.df_final.ix['x'], train_label], axis=1)
			df_test = pd.concat([test_index, self.df_final.ix['y']], axis=1)
			df_train.to_csv(path_train, index = index)
			df_test.to_csv(path_test, index = index)
#			self.df_final.to_csv(path, index = index)
		else:
			self.aggregate_features()
			self.to_csv(path_train, path_test, train_index, train_label, test_index, index)
#			raise ValueError('Call method aggregate_features before call this method')

class local_evaluation(object):
	@classmethod
	def bi_classification_4(cls, y_true, y_pred, y_pred_prob = None):
		from sklearn import metrics
		recall = metrics.recall_score(y_true, y_pred)
		precision = metrics.precision_score(y_true, y_pred)
		f1 = metrics.f1_score(y_true, y_pred)
		if y_pred_prob is not None:
			auc = metrics.roc_auc_score(y_true, y_pred_prob)
		else:
			auc = metrics.roc_auc_score(y_true, y_pred)
		print "Recall is %f, Precision is %f, f1 score is %f, and AUC is %f" % (recall, precision, f1, auc)

class lookup(read_write):
	def dict_create(self, keys, prefix = None, file_path = None, shuff = False): # keys is an array of string
		if shuff:
			np.random.shuffle(keys) # random shuffle the array
		if prefix:
			values = map(lambda x: prefix.format(x), xrange(1,len(keys)+1))
		else:
			values = xrange(1,len(keys)+1)
		list_of_turples = zip(keys, values)
		dictionary = dict(list_of_turples)
		if file_path:
			read_write().array_to_csv(list_of_turples, file_path, np.array(['orginal_id', 'new_id']))
		return dictionary

	def dict_validate(self, origin, descent, dictionary, cln_name):
		'''
		This method take 3 files as input: origin, descent and dictionary, and then check whether the lookup is totally correct;
		Here we assume that the order of row does not change after lookup
		'''
		(header, arr_origin) = self.tsv_to_array(origin)
		(header, arr_descent) = self.csv_to_array(descent, "rU")
		cln_id = np.where(header == cln_name)[1][0]
		print cln_id
		(header, arr_dict) = self.csv_to_array(dictionary)
		dict_dict = dict(zip(arr_dict[:, 0], arr_dict[:, 1]))
		for row_origin, row_descent in zip(arr_origin, arr_descent):
			if dict_dict[row_origin[cln_id]] != row_descent[cln_id]:
				print "Error"
				print "Original id is %s, descent id is %s: " % (row_origin[cln_id], row_descent[cln_id])
				print "The dictionary of original id %s is %s" % (row_origin[cln_id], dict_dict[row_origin[cln_id]])
				print

	def dict_validate_II(self, origin, descent, dictionary, country_ctl = False):
		'''
		This method take 2 files as input: origin and descent; and a dictionary of dictionaries of all masked IDs, 
		and then check whether the lookup is totally correct;
		Here we assume that the order of row does not change after lookup
		'''
		(header_1, arr_origin) = self.tsv_to_array(origin)
		(header_2, arr_descent) = self.csv_to_array(descent)
		for cln_name in header_1.flatten():
			if country_ctl:
				true_false = (cln_name in dictionary.keys()) & (cln_name != "country")
			else:
				true_false = cln_name in dictionary.keys()
			if true_false:
				dict_dict = dictionary[cln_name] # This is to extract the dictionary for this particular column
				cln_id_1 = np.where(header_1 == cln_name)[1][0]
				cln_id_2 = np.where(header_2 == cln_name)[1][0]
				print cln_name
				print cln_id_1
				print cln_id_2
				for row_origin, row_descent in zip(arr_origin, arr_descent):
					if dict_dict[row_origin[cln_id_1]] != row_descent[cln_id_2]:
						print "Error"
						print "Original id is %s, descent id is %s: " % (row_origin[cln_id_1], row_descent[cln_id_2])
						print "The dictionary of original id %s is %s" % (row_origin[cln_id_1], dict_dict[row_origin[cln_id_1]])
						print


class slicing(object):
	def chunks(self, array, granular, arr_len):
		for i in xrange(0, arr_len, granular):
			yield array[i:i+granular]

	def chunks_arr(self, array, granular_array):
		index = 0
		for i in xrange(0, len(granular_array)):
			yield array[index : index + granular_array[i]]
			index = index + granular_array[i]

class sort_arr(object):
	def one_index_sort(self, array, cln):
		return array[array[:, cln].astype(int).argsort(),:]

	def two_index_sort(self, array, cln_1, cln_2, reverse_1 = False, reverse_2 = False):
		order_1 = 1
		order_2 = 1
		if reverse_1:
			order_1 = -1
		if reverse_2:
			order_2 = -1
		ind = np.lexsort((array[:, cln_2].astype(int) * order_2, array[:, cln_1].astype(int) * order_1))
		return array[ind,:]

class stratification(object):
	@classmethod
	def one_cln2(cls, array, train = 0.8): #array is the 1-D array -> the column to be stratified
		random_state = 8#int(time.time()%60)
		s3 = sss(array, n_iter = 1, test_size = (1.0 - train), random_state = random_state)
		print 'random_state is ', random_state
		for train_index, test_index in s3:
			pass
		return (train_index, test_index)

	@classmethod
	def one_cln3(cls, array, train = 0.6, cross = 0.2): #array is the 1-D array -> the column to be stratified
		s3 = sss(array, n_iter = 1, test_size = (1.0 - train - cross), train_size = train, random_state = int(time.time()%60))
		for train_index, test_index in s3:
			pass
		index = np.array(xrange(array.size))
		cross_index = np.setdiff1d(index, np.concatenate((train_index, test_index)), assume_unique = True)
		np.random.shuffle(cross_index)
		return (train_index, cross_index, test_index)

	# The method below select train/test/validation datasets based on stratification on one column
	# It returns a turple of 3 arrays containing the index for train/test/validation datasets respectively
	def one_cln(self, array, train = 0.6, cross = 0.2): #array is the 1-D array -> the column to be stratified
		s3 = sss(array, n_iter = 1, test_size = (1.0 - train - cross), train_size = train, random_state = int(time.time()%60))
		for train_index, test_index in s3:
			pass
		index = np.array(xrange(array.size))
		cross_index = np.setdiff1d(index, np.concatenate((train_index, test_index)), assume_unique = True)
		np.random.shuffle(cross_index)
		return (train_index, cross_index, test_index)

	def two_cln_fake(self, header, array, clns, train = 0.6, cross = 0.2):
		pass

	def data_randomnize(self, data_extracted, split_ratio): # This function will randomnize the given dataset, and then split them according the split ratio
		np.random.shuffle(data_extracted)
		num_rows = np.size(data_extracted[:,0])
		num_row_train = int(round(num_rows*split_ratio[0]))
		num_row_cross = int(round(num_rows*split_ratio[1]))
	#	num_row_test = num_rows - num_row_train - num_row_cross
		return (data_extracted[0:num_row_train,:], data_extracted[num_row_train:(num_row_train + num_row_cross),:], data_extracted[(num_row_train + num_row_cross):num_rows,:])
		print num_rows

	def two_clns(self, data_header, data_whole, clns, split_ratio = [0.6, 0.2, 0.2]):
		'''
		This method is to handle data stratification in two columns;
		data_header is the 2-D array of header names, data_whole is the 2-D array of whole data
		clns is the list of the names of 2 columns to be splitted based on
		split_ratio is the list of ratios of train, cross validation and test datasets
		'''
	## Step 1. Find the index number of the columns according to their names	
		stratified_cln = [0,1]
		i = 0
		for cln_name in clns:
			stratified_cln[i] = np.where(data_header == cln_name)[1][0]
#!			print np.where(data_header == cln_name)
			i = i + 1
#!		print stratified_cln
	## Step 2. Create two lists, one is the list of lists of unique values of each of 2 clns, one is the list of # of unique values in each of 2 clns
		num_rows = np.size(data_whole[:,0])
		stratified_cln_unique_value = []
		temp_list = []
		for i in stratified_cln:
			temp_array = np.unique(data_whole[:,i])
			temp_list.append(list(temp_array))
			stratified_cln_unique_value.append(len(temp_array))
	#!	print temp_list[0][0]
	## Step 3. Segment the rows in to (m X n) segments, here m, n are # of unique values of in each of 2 clns
		data_train_array = data_header
		data_cross_array = data_header
		data_test_array = data_header
		for i in xrange(stratified_cln_unique_value[0]):
			for j in xrange(stratified_cln_unique_value[1]):
				data_extracted = data_whole[(data_whole[:,stratified_cln[0]] == temp_list[0][i]) & (data_whole[:,stratified_cln[1]] == temp_list[1][j]),:]
				(temp_train, temp_cross, temp_test) = self.data_randomnize(data_extracted, split_ratio)
	#!			print temp_train
				data_train_array = np.vstack((data_train_array, temp_train))
				data_cross_array = np.vstack((data_cross_array, temp_cross))
				data_test_array = np.vstack((data_test_array, temp_test))
	#!			print data_extracted
		return data_train_array, data_cross_array, data_test_array



class recommendation(object): # This class contains methods used for recommendation system
	def noise_add(self, percentage): # This method is to add additional noises to the cross validation and test datasets, and those noise user-movie pairs won't be evaluated
		pass

class rakuten(slicing, sort_arr, lookup, read_write): # Import Derived class first, and then the base class. Here loopup is derived class of read_write. Actually no need to inherit 'read_write'
	def function():
		pass

class mindef(data_clean, read_write, sort_arr, stratification, create_files_for_participant, evaluation_metric):
	def function():
		pass

class mindef_benchmark(data_clean, read_write):
	def function():
		pass