#ifndef _DTW_R_CLASSIFIER_H
#define _DTW_R_CLASSIFIER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "example.h"
#include "util.h"

using namespace std;

class DTW_R_example {
   vector <double> v; // time series
   string label;
   string name; // dataset name
   int id; // index of the example in the data file
   bool ifTrain; // true if train example

   public:
   DTW_R_example() {}

   DTW_R_example(const example &pe) {
      v = pe.giveV();
      label = pe.giveLabel();
      name = pe.giveName();
      id = pe.giveId();
      ifTrain = pe.giveIfTrain();
   }

   double dtw_r(const DTW_R_example &o, int R) const;
   double compute_dtw_r(const DTW_R_example &o, int R) const;

   string giveLabel() const {
      return label;
   }
};

class DTW_R_classifier {
      // ptr: training examples, pte: test example,; R: parameter, label: output labels
      double train_test(const vector <DTW_R_example> &ptr, const vector <DTW_R_example> &pte, int R, vector <string> &label); 
   public: 
      double cross_validation(const vector <example> &ptr, int R, int folds);
      double train_test(const vector <example> &ptr, const vector <example> &pte, const vector <int> &pR); 
      // pR: list of values for parameter R; bR: best value of R found through cross-validation  
      double train_test(const vector <example> &ptr, const vector <example> &pte,  const vector <int> &pR, int &bR, vector <string> &label, double &cv_error); 
};

#endif
