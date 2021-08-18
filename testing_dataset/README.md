# Testing dataset directory
# putting your testing dataset here
- [x] [VOT2018](http://www.votchallenge.net/vot2018/dataset.html)
- [x] [VOT2019](https://www.votchallenge.net/vot2019/dataset.html)
- [x] [OTB100(OTB2015)](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html)
- [x] [UAV123](https://ivul.kaust.edu.sa/Pages/Dataset-UAV123.aspx)
- [x] [NFS](http://ci2cv.net/nfs/index.html)
- [x] [LaSOT](https://cis.temple.edu/lasot/)

## Download Dataset
Download [json files](https://drive.google.com/drive/folders/10cfXjwQQBQeu48XMf2xc_W1LucpistPI).

1. Put CVRP13.json, OTB100.json, OTB50.json in OTB100 dataset directory (you need to copy Jogging to Jogging-1 and Jogging-2, and copy Skating2 to Skating2-1 and Skating2-2 or using softlink)

   The directory should have the below format

   | -- OTB100/

   ​	| -- Basketball

   ​	| 	......

   ​	| -- Woman

   ​	| -- OTB100.json

   ​	| -- OTB50.json

   ​	| -- CVPR13.json

2. Put all other jsons in the dataset directory like in step 1
