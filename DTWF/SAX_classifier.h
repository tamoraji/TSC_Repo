#ifndef _SAX_CLASSIFIER_H
#define _SAX_CLASSIFIER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include "example.h"

using namespace std;

class SAX_example {
   vector <double> v; // time series
   
   unordered_map <int,double> fv; // hash to store features and their values
   // the word is converted into a number which forms the key for the hash
   
   string label;
   string name; // dataset name
   int id; // index of the example in the data file
   bool ifTrain; // true if a train example
   void makeSAXfeatures(int n, int w, int a);

   public:

   SAX_example(const example &pe, int n, int w, int a) {
      v = pe.giveV();
      label = pe.giveLabel();
      name = pe.giveName();
      id = pe.giveId();
      ifTrain = pe.giveIfTrain();
      makeSAXfeatures(n,w,a);
   }

   string giveLabel() const {
      return label;
   }

   double featureEuclideanDistance(const SAX_example &o) const;
   double featureDotProduct(const SAX_example &o) const;
   double featureCosine(const SAX_example &o) const;

};

class SAX_classifier {
      // ptr: training examples, pte: test example,; label: output labels
      double train_test(const vector <SAX_example> &ptr, const vector <SAX_example> &pte, vector <string> &label); 
   public: 
      // pn,pw,pa: parameter values
      double cross_validation(const vector <example> &ptr, int pn, int pw, int pa, int folds);

      // pn: list of values for parameter n; bn: best value of n found through cross-validation  
      // pw: list of values for parameter w; bw: best value of w found through cross-validation  
      // pa: list of values for parameter a; ba: best value of a found through cross-validation  
      double train_test(const vector <example> &ptr, const vector <example> &pte, const vector <int> &pn, const vector <int> &pw, const vector <int> &pa, int &bn, int &bw, int &ba, vector <string> &label); 
      double train_test(const vector <example> &ptr, const vector <example> &pte, const vector <int> &pn, const vector <int> &pw, const vector <int> &pa); 
};

#endif
