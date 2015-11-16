# About
Including the evaluation metric Eexpected Weighted Average Precision (EWAP) used in Rakuten-Viki Data Analytics challenge in challenges.dextra.sg. The codes are written in both python and R, but the acutally implementation in DEXTRA platform is done in php, which is maintained by Aung


## Introduction

### EWAP.py
Implementation of EWAP in python.

### EWAP.R
Implementation of EWAP in R.

### predicted.csv
The sample data that emulates participants' submission. It is used to test the metric.

### true.csv
The sample data that emulates the ground truth of submission hidden from participants. It is usd together with predicted.csv to test the metric.


## How to run?

### Run in Python

*1.* Download this folder to your local directory.

*2.* Put [DataProcessing.py](https://github.com/newtoncircus/DEXTRA_SourceCodes/blob/master/DataProcessing.py) in this folder.

*3.* In the Terminal, type python /path/to/this/folder/EWAP.py.

### Run in R

*1.* Download this folder to your local directory.

*2.* Open [EWAP.R](https://github.com/newtoncircus/DEXTRA_SourceCodes/blob/master/EWAP/EWAP.R) in R Studio.

*3.* In the code, change the file_path to the desired path in your local directory.

*4.* Run the code in R studio.