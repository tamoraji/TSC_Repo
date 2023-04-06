#ifndef _SVM_CLASSIFIER_H
#define _SVM_CLASSIFIER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include "example.h"
#include "SAX_classifier.h"

using namespace std;

class SVM_example {
   vector <double> v; // time series
   string label;
   string name; // dataset names
   int id; // index of the example in the data file
   bool ifTrain; // true if train example
   bool fEuclidean, fSAX, fDTW, fDTW_R; // whether has those features 
   
   vector <double> feature_Euclidean, feature_DTW, feature_DTW_R; // features
   unordered_map <int,double> fv; // SAX features and values in a hash

   double euclidean_distance(const SVM_example &o) const;
   double dtw(const SVM_example &o) const;
   double compute_dtw(const SVM_example &o) const;
   double dtw_r(const SVM_example &o, int R) const;
   double compute_dtw_r(const SVM_example &o, int R) const;

   string ktype; // kernel type: linear, poly2, poly3, rbf

   public:

   SVM_example() {}

   SVM_example(const example &pe) {
      v = pe.giveV();
      label = pe.giveLabel();
      name = pe.giveName();
      id = pe.giveId();
      ifTrain = pe.giveIfTrain();
      fEuclidean=false;
      fSAX=false;
      fDTW=false;
      fDTW_R=false;
   }

   string giveLabel() const {
      return label;
   }

   void setK(const string &k) {
      ktype = k;
   }


   void addEuclideanfeatures(const vector <SVM_example> &tr);
   void addSAXfeatures(int n, int w, int a);
   void addDTWfeatures(const vector <SVM_example> &tr);
   void addDTW_Rfeatures(const vector <SVM_example> &tr, int R);

   double kernelUn(const SVM_example &o) const;
};

class SVM_classifier {
   // ptr: training examples, pte: test example,; label: output labels
   void svm_train(const vector <SVM_example> &ptr);
   void svm_test(const vector <SVM_example> &ptr, const vector <SVM_example> &pte, vector <string> &label);
   double train_test(const vector <SVM_example> &ptr, const vector <SVM_example> &pte, vector <string> &label); 
   public: 
      // bool variables: if those features should be used
      // pn,pw,pa: SAX parameter values; R: DTW_R parameter value; k: kernel
      double cross_validation(const vector <example> &ptr, bool pfEuclidean, bool pfSAX, bool pfDTW, bool pfDTW_R, int pn, int pw, int pa, int R, const string &k, int folds);
      // pn: list of values for parameter n; bn: best value of n found through cross-validation  
      // pw: list of values for parameter w; bw: best value of w found through cross-validation  
      // pa: list of values for parameter a; ba: best value of a found through cross-validation  
      // pR: list of values for parameter R; bR: best value of R found through cross-validation  
      // pk: list of values for kernel k;    bk: best value of k found through cross-validation  
      double train_test(const vector <example> &ptr, const vector <example> &pte, bool pfEuclidean, bool pfSAX, bool pfDTW,  bool pfDTW_R, const vector <int> &pn, const vector <int> &pw, const vector <int> &pa, const vector <int> &pR, const vector <string> &pk, int &bn, int &bw, int &ba, int &bR, string &bk, vector <string> &label, double &cv_error); 
      double train_test(const vector <example> &ptr, const vector <example> &pte, bool pfEuclidean, bool pfSAX, bool pfDTW,  bool pfDTW_R, const vector <int> &pn, const vector <int> &pw, const vector <int> &pa, const vector <int> &pR, const vector <string> &pk); 
};

#endif
