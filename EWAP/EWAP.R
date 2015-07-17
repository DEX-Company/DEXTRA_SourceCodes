WAP <- function(num, predicted, actual, actual_score, n) # This is function to calculate [WAP@num] for each user
{
	sum_actual_score <- 0.0
	sum_predicted_score <- 0.0
	sum_ratio <- 0.0
	for (i in 1:num)
	{
		if (predicted[i] %in% actual)
		{
			predicted_score <- actual_score[which(actual %in% predicted[i])] # This step may require a loop to find the corresponding actual_score
		}
		else
		{
			predicted_score <- 0.0
		}
		sum_predicted_score <- sum_predicted_score + predicted_score
		if (i <= n)
		{
			sum_actual_score <- sum_actual_score + actual_score[i]
		}
		sum_ratio <- sum_ratio + sum_predicted_score/sum_actual_score
	}
	sum_ratio/num
}

## Step 1. Set up the working directory, read the sample files into data frame
file_path <- "../DEXTRA_SourceCodes/EWAP"
setwd(file_path)

predicted_path <- "predicted.csv"
true_path <- "true.csv"

data_predicted <- read.csv(predicted_path)
#View(sample_submission)

data_true <- read.csv(true_path)
#View(pub_validation)

## Step 2. Convert the 2 data frames into 3 hashtables (i.e. dictionaries)
## Step 2.1 create the dictionary of user_id and the videos the user actually watched
video_true <- aggregate(video_id ~ user_id, data_true, function(x) as.vector(x))
dict_video_true <- video_true$video_id
names(dict_video_true) <- video_true$user_id

## Step 2.2 create the dictionary of user_id and the score of videos the user actually watched
score_true <- aggregate(score ~ user_id, data_true, function(x) as.vector(x))
dict_score_true <- score_true$score
names(dict_score_true) <- score_true$user_id

## Step 2.3 create the dictionary of user_id and the videos recommended to the user
video_predicted <- aggregate(video_id ~ user_id, data_predicted, function(x) as.vector(x))
dict_video_predicted <- video_predicted$video_id
row.names(dict_video_predicted) <- video_predicted$user_id

if (! all.equal(video_predicted$user_id, video_true$user_id))
{
  print("Error! users in predicted data do not match actual data!")
}

## Step 3 Loop through all users, calculate weighted average precision (wap), and finally EWAP 
Score_of_all_users <- 0.0
EWAP <- 0.0
num <- 3
for (user in video_true$user_id)
{
  key <- toString(user)
	predicted <- dict_video_predicted[key,] # Note: only 3 videos
	actual <- dict_video_true[key][[1]] # Note: contains at least 1 videos
	actual_score <- dict_score_true[key][[1]] # Note: the videos in actual list is sorted from highest score to lowest
	idx <- order(-actual_score) # return the index of descending sort on score
  actual_score <- actual_score[idx]
  actual <- actual[idx]
#  print(predicted)
#  print(actual)
#  print(actual_score)
#  print("-----------")
  m <- length(actual)
	n <- min(num, m)
	S_user <- sum(as.vector(actual_score))
	Score_of_all_users <- S_user + Score_of_all_users
	wap_user <- WAP(num, predicted, actual, actual_score, n)
	EWAP <- EWAP + wap_user * S_user
}
EWAP_final <- EWAP / Score_of_all_users