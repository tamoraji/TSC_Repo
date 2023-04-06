#ifndef _EXPERIMENT_H
#define _EXPERIMENT_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>
#include <cstdlib>

#include "example.h"

using namespace std;

class classifier {
   public:
   string type; // type of classifier
   string outdir; // output directory
   // parameters (includes for all types of classifiers)
   vector <int> kval, n, bin, agg, w , a, r;
   vector <double> lambda;
   vector <string> k, features;
};

class experiment {
   string name;
   vector <example> train, test;

   vector <string> labelMap;

   public:

   experiment() {
   }

   experiment(const string &pname) {
      name = pname;
   }

   string labelName(int i) const {
      assert(i < labelMap.size());
      return labelMap[i];
   }
   
   string labelName(const string &s) const {
      int i = atoi(s.c_str());
      return labelName(i);
   }
   
   string giveName() const {
      return name;
   }

   int trainSize() const {
      return train.size();
   }

   int testSize() const {
      return test.size();
   }

   // read examples from a file
   void readExamples(const string &trainFile, const string &testFile); 

   double Euclidean_train_test(ostream &out) const;
   // n,w,a: list of values for SAX features 
   double SAX_train_test(const vector <int> &n, const vector <int> &w, const vector <int> &a, ostream &out) const;
   double DTW_train_test(ostream &out) const;
   double DTW_R_train_test(const vector <int> &r, ostream &out) const;
   // bools tell which features to use, n,w,a,r,k(kernel): list of values for parameters  
   double SVM_train_test(bool fEuclidean, bool fSAX, bool fDTW, bool fDTW_R, const vector <int> &n, const vector <int> &w, const vector <int> &a, const vector <int> &r, const vector <string> &k, ostream &out) const;
   ostream& write(ostream &out) const;
};


#endif
