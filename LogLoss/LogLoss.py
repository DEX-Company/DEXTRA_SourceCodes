import numpy as np
import math

def LogLoss(truth, prediction):
	'''
	This method take 2 inputs: the array of true value and the array of predicted value; and the the method will evaluate the prediction against
	truth using Log Loss metric
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

def main():
	'''
	This is to demostrate how the evaluation metric works using sample inputs
	'''
	truth = np.array([['ID001', '1'], ['ID002', '0'], ['ID003', '0'], ['ID004', '0'], ['ID005', '1'], ['ID006', '0'], ['ID007', '1']])
	prediction = np.array([['ID001', '0.9'], ['ID002', '0.01'], ['ID003', '0'], ['ID004', '1'], ['ID005', '0'], ['ID006', '0.99'], ['ID007', '0.99']])
	print LogLoss(truth, prediction)

if __name__ == "__main__":
	main()