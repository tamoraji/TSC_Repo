#include <cassert>
#include <cmath>
#include "Euclidean_classifier.h"

double Euclidean_example::distance(const Euclidean_example &o) const {
   //assert(v.size()==o.v.size());
   double r=0;
   int max; // maximum index to consider in case they are not equal 
   if (v.size() > o.v.size()) max = o.v.size();
   else max = v.size();
   for (int i=0; i<max; ++i) {
      r += pow((v[i]-o.v[i]),2);
   }
   return sqrt(r);
}

double Euclidean_classifier::train_test(const vector <Euclidean_example> &ptr, const vector <Euclidean_example> &pte, vector <string> &label) {
   assert(ptr.size() > 0);

   label.resize(pte.size());
   vector <string> corr_label;
   // one nearest neighbor classification 
   for (int i=0; i<pte.size(); ++i) {
      corr_label.push_back(pte[i].giveLabel());
      double minD=ptr[0].distance(pte[i]);
      int c=0;
      for (int j=1; j<ptr.size(); ++j) {
	 double d=ptr[j].distance(pte[i]);
	 if (d < minD) {
	    minD = d;
	    c=j;
	 }
      }
      label[i] = ptr[c].giveLabel();
   }
   return compute_error(corr_label,label);
}

double Euclidean_classifier::cross_validation(const vector <example> &ptr, int folds) {
   assert(ptr.size() > 0);
      
   vector <Euclidean_example> tr;
   for (int i=0; i<ptr.size(); ++i) {
      tr.push_back(Euclidean_example(ptr[i]));
   }

   vector <string> outLabel, corrLabel;
   for (int f=0; f<folds; ++f) {
      // make train-test examples for the fold
      vector <Euclidean_example> ftr,fte;
      for (int i=0; i<tr.size(); ++i) {
	 if (i % folds == f) {
	    fte.push_back(tr[i]);
	    corrLabel.push_back(tr[i].giveLabel());
	 }
	 else ftr.push_back(tr[i]);
      }
      vector <string> label;
      train_test(ftr,fte,label);
      for (int i=0; i<label.size(); ++i) outLabel.push_back(label[i]);
   }

   return compute_error(corrLabel,outLabel);
}

double Euclidean_classifier::train_test(const vector <example> &ptr, const vector <example> &pte) {
   vector <string> label;
   return train_test(ptr,pte,label);
}

double Euclidean_classifier::train_test(const vector <example> &ptr, const vector <example> &pte, vector <string> &label) {
   assert(ptr.size() > 0);
   vector <Euclidean_example> tr, te;
   for (int i=0; i<ptr.size(); ++i) {
      tr.push_back(Euclidean_example(ptr[i]));
   }
   for (int i=0; i<pte.size(); ++i) {
      te.push_back(Euclidean_example(pte[i]));
   }
   return train_test(tr,te,label);
}
