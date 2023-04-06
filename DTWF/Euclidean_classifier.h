#ifndef _EUCLIDEAN_CLASSIFIER_H
#define _EUCLIDEAN_CLASSIFIER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "example.h"
#include "util.h"

using namespace std;

class Euclidean_example {
   vector <double> v; // time series
   string label;
   string name; // dataset name
   int id; // index of the example in the data file
   bool ifTrain; // true if train example

   public:
   Euclidean_example(const example &pe) {
      v = pe.giveV();
      label = pe.giveLabel();
      name = pe.giveName();
      id = pe.giveId();
      ifTrain = pe.giveIfTrain();
   }

   double distance(const Euclidean_example &o) const;

   string giveLabel() const {
      return label;
   }
};

class Euclidean_classifier {
      // ptr: training examples, pte: test examples, label: output labels
      double train_test(const vector <Euclidean_example> &ptr, const vector <Euclidean_example> &pte, vector <string> &label); 
   public: 
      double cross_validation(const vector <example> &ptr, int folds);
      double train_test(const vector <example> &ptr, const vector <example> &pte, vector <string> &label); 
      double train_test(const vector <example> &ptr, const vector <example> &pte); 
};

#endif
