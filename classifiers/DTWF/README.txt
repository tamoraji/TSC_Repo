RADME.txt

This is the code for Time Series Classification by Rohit J. Kate (katerj@uwm.edu) 
correponding to the paper "Using Dynamic Time Warping Distances as Features for Improved 
Time Series Classification", Rohit J. Kate, Data Mining and Knowledge Discovery, May 2015.

The code is in C++ and it has been used and tested only on Linux.

LICENSE
-------

UWM Time Series Classification Code
Copyright (C) 2014 Rohit J. Kate

This program is free software: you can redistribute it and/or modify it 
under the terms of the GNU General Public License as published by the 
Free Software Foundation, either version 3 of the License, or (at your option) 
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
this program. If not, see<http://www.gnu.org/licenses/>.


Install
-------

1. Download libsvm from http://www.csie.ntu.edu.tw/~cjlin/libsvm/
   Compile it so that its  executables svm-train and svm-predict 
   are available in a directory. Include the global path of that directory 
   as the value for the string variable LIBSVMDIR at the beginning in the file 
   SVM_classifier.cpp. For the results in the paper, version 3.17 of libsvm was used. 

2. Compile the code by doing "make". The code is known to compile with g++ version 4.4.7.
   It will generate executable main.

3.  Compile the code for ensembles by "g++ -O3 ensemble.cpp -o ensemble" 

Run
---

1. Prepare an experiment list file. A few sample ones are shown in the directory expt_list.
   The file requires xml format (with space separation between tags and entities).
   First all the classifiers to be experimented are listed and then all the datasets 
   are listed. Within the space of classifiers, the output directory is specified 
   along with the list of parameters appropriate for a classifier. If multiple values
   of classifiers are specified then the program finds the best value out of them
   through internal cross-validation within the training data. The type of classifier
   can be: Euclidean, DTW, DTW_R, SAX or SVM. All the feature-based versions are run under
   SVM classifier. Please see the parameters needed for the different classifiers in
   their respective sample expt_list files. In addition, for SVM classifier, the parameters for
   DTW_R or SAX features for individual datasets need to be specified in the dataset fields.
   Note that the training and testing files for the datasets need to be specified in the 
   dataset fields. They should be in the same format as comes with the UCR datasets.
   
2. Run the program by "./main [-cache] expt_list_file". The output (accuracies will go in the
   output directories as specified in expt_list file, create the outermost directories in advance.).
   An option to cache all the DTW or DTW_R computations is provided with the option of "-cache".
   This will compute all the DTW and DTW_R computations needed for the expt_list file
   and store them in the directory "cache/" if they have not been already computed. 
   Run the program afterwards without the "-cache" option (this will run much faster).

3. To create ensemles, first the component classifiers must have been run
with internal cross-validation option of "./main -cv expt_list_file" (if there 
were more than one parameter combinations, then internal cross-validation will happen
even without -cv option). Next, run "./ensemble outdir dir1 dir2 dir3 .." where dirs
are the directories where the outputs of component classifiers are present and
the output of the ensemble will go in outdir.


Comparison
----------

1. Install Python 2.7 with scipy library.
2. Run the program wilcoxon.py to compare results in two output directories
   by doing "import wilcoxon", followed by "wilcoxon.main("dir1","dir2")".
   It will give the win/loss/tie as well as p-value for Wilcoxon's signed
   ranked test.


For any questions and comments, please contact Rohit Kate (katerj@uwm.edu).
