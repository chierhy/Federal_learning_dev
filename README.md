# Federal_learning_dev

Federal learning 2.1
created by Xiao Peng, Lin Ze, Hao Ying, Dr. Wang Chuang

fault diagnosis.py is the main processing file for "overall model"
xiao_fault diagnosis.py is the main processing file for "vote model"

model.py define several CNN model, 
	which "CNN2d_classifier_xiao" is used for clip size=2304;
	"CNN2d_classifier" is used for clip size=4096 (only for previous data precessing methods);
	"CNN2d_fitting_xiao" is used for clip size=576;
	other model is not used yet.

xiao_estimate.py is used for vote model estimate, it load each part-model and test accuracy of vote model.

xiaodataset.py is for data process, compare to the previous methods, it can increase the utilization rate of data.
xiao_dataset_random.py is for data process too, this can divide the data into train and test randomly.
