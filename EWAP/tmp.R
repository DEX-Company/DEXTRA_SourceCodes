video_true <- aggregate(video_id ~ user_id, data_true, function(x) as.vector(x))
dict_video_true <- video_true$video_id
row.names(dict_video_true) <- video_true$user_id