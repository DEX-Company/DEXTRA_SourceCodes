#! This script is to demostrate how the evaluation metric Expected Weighted Average Precision works in python
import numpy as np
from itertools import groupby

def wap(num, predicted, actual, score, n):
	video_score_dict = dict(zip(actual, score))
	sum_actual_score_denominator = 0.0
	sum_predicted_score_numerator = 0.0
	sum_ratio = 0.0
	for i in xrange(num):
		if predicted[i] in actual:
			predicted_score = float(video_score_dict[predicted[i]])
		else:
			predicted_score = 0.0
		sum_predicted_score_numerator = sum_predicted_score_numerator + predicted_score
		if i < n:
			sum_actual_score_denominator = sum_actual_score_denominator + float(score[i])
		sum_ratio = sum_ratio + sum_predicted_score_numerator/sum_actual_score_denominator
	return sum_ratio/num



def EWAP(arr_predicted, arr_actual, num):
	'''
	This is the sample code to calculate the evaluation metric EWAP @ num.
	it takes 3 input arguments: the 2 columns array of prediction -> arr_predicted, the 3 columns array of ground truth -> arr_actual, and num is the cut-off number for the metric
	'''
## Step 0. Set the values for some parameters
	arr_predicted_len = arr_predicted[:,0].size
	# The below are three columns from actual array
	cln_id_user_actual = 0
	cln_id_video_actual = 1
	cln_id_score_actual = 2
	# The below are two columns from predicted array
	cln_id_user_predicted = 0
	cln_id_video_predicted = 1

## Step 1. Check whether the two arrays have the same number of unique users
	uni_actual, idx, user_granular = np.unique(arr_actual[:, cln_id_user_actual].astype(int), return_index = True, return_counts = True)
	uni_predicted = np.unique(arr_predicted[:, cln_id_user_predicted].astype(int))
	if not np.array_equal(uni_actual, uni_predicted):
		print "Error, Inconsistent number of users between prediction array and actual array"

## Step 2. Sort the array of actul values according to user_id column (ascending) and then score column (descending)
	ind = np.lexsort((- arr_actual[:, cln_id_score_actual].astype(int), arr_actual[:, cln_id_user_actual].astype(int)))
	arr_actual = arr_actual[ind, :]
## Step 2.1 Create the dictionary from the sorted actual array; the dictionary will be used to loop through, using user_id as the key
	dict_arr_actual = {}
	for key, group in groupby(arr_actual, key = lambda x: x[cln_id_user_actual]):
#		print key, ([v[1] for v in group], [v[2] for v in group])
		list_video = []
		list_score = []
		for v in group:
			list_video.append(v[cln_id_video_actual])
			list_score.append(int(v[cln_id_score_actual]))
		dict_arr_actual.update({key: (list_video, list_score)})
#	print dict_arr_actual
#	print
## Step 3. Create the dictionary from the predicted array
	dict_arr_predicted = {}
	for key, group in groupby(arr_predicted, key = lambda x: x[cln_id_user_predicted]):
		dict_arr_predicted.update({key: [v[1] for v in group]})
#	print dict_arr_predicted
## Step 4. Loop through the two dictionaries created above, calculate the value wap
	score_of_all_users = 0.0
	ewap = 0.0
	for user in dict_arr_actual.keys():
		list_video_actual = dict_arr_actual[user][0]
		list_score_actual = dict_arr_actual[user][1]
		list_video_predicted = dict_arr_predicted[user]

		m = len(list_video_actual)
		n = min(num, m)
		s_user = sum(list_score_actual)
		score_of_all_users = score_of_all_users + s_user

		wap_user = wap(num, list_video_predicted, list_video_actual, list_score_actual, n)
		ewap = ewap + wap_user * s_user
		'''
		print wap_user
		print s_user
		print list_video_actual
		print list_score_actual
		print list_video_predicted
		print m
		print
		'''
		
	ewap_final = ewap / score_of_all_users
	print ewap_final
	print score_of_all_users


def main():
	import DataProcessing as dp # This is my customized module for data pre-process, can be found here : https://github.com/newtoncircus/DEXTRA_SourceCodes/blob/master/DataProcessing.py
	root_path = "./"
#	path_predicted = "SampleSubmission_percent10.csv"
#	path_actual = "public_validation_datasheet_percent10.csv"
	path_predicted = "predicted.csv"
	path_actual = "true.csv"
	viki = dp.rakuten()
	(header_sample, arr_predicted) = viki.csv_to_array(root_path + path_predicted, "rU")
	(header_public, arr_actual) = viki.csv_to_array(root_path + path_actual, "rU")
	num = 3
	EWAP(arr_predicted, arr_actual, num)


if __name__ == "__main__":
	main()