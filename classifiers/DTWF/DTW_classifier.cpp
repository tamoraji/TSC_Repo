#include <cassert>
#include <cmath>
#include <cfloat>
#include "util.h"
#include "DTW_classifier.h"
#include "cache.h"

extern cache C;
extern bool CV;

double DTW_example::dtw(const DTW_example &o) const {
   // dtw distance, first checks the cache otherwise computes
   double r;
   vector <int> iid;
   if (ifTrain) {
      if (o.ifTrain) iid.push_back(0);
      else iid.push_back(1);

      double r;
      if (C.giveValue(vector <string>(1,name),iid,id,o.id,r)) {
	 return r;
      }
   }
 
   // not in cache
   return compute_dtw(o);
}

double DTW_example::compute_dtw(const DTW_example &o) const {
   vector <vector <double> > d(v.size()+1, vector <double> (o.v.size()+1));
   for (int i=1; i<=v.size(); ++i) {
      d[i][0] = DBL_MAX;
   }
   for (int j=1; j<=o.v.size(); ++j) {
      d[0][j] = DBL_MAX;
   }
   d[0][0] = 0;
   for (int i=1; i<=v.size(); ++i) {
      for (int j=1; j<=o.v.size(); ++j) {
         double min=d[i-1][j];
         if (min > d[i][j-1]) min=d[i][j-1];
         if (min > d[i-1][j-1]) min=d[i-1][j-1];
         d[i][j] = pow((v[i-1]-o.v[j-1]),2) + min;
      }
   }
   return sqrt(d[v.size()][o.v.size()]);
}

double DTW_classifier::train_test(const vector <DTW_example> &ptr, const vector <DTW_example> &pte, vector <string> &label) {
   assert(ptr.size() > 0);

   label.resize(pte.size());
   vector <string> corr_label;
   // one nearest neighbor classification using DTW 
   for (int i=0; i<pte.size(); ++i) {
      corr_label.push_back(pte[i].giveLabel());
      double minD=ptr[0].dtw(pte[i]);
      int c=0;
      for (int j=1; j<ptr.size(); ++j) {
	 double d=ptr[j].dtw(pte[i]);
	 if (d < minD) {
	    minD = d;
	    c=j;
	 }
      }
      label[i] = ptr[c].giveLabel();
   }
   return compute_error(corr_label,label);
}

double DTW_classifier::cross_validation(const vector <example> &ptr, int folds) {
   assert(ptr.size() > 0);
      
   vector <DTW_example> tr;
   for (int i=0; i<ptr.size(); ++i) {
      tr.push_back(DTW_example(ptr[i]));
   }

   vector <string> outLabel, corrLabel;
   for (int f=0; f<folds; ++f) {
      // make train-test examples for the fold
      vector <DTW_example> ftr,fte;
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

double DTW_classifier::train_test(const vector <example> &ptr, const vector <example> &pte) {
   vector <string> label;
   double cv_error;
   return train_test(ptr,pte,label,cv_error);
}

double DTW_classifier::train_test(const vector <example> &ptr, const vector <example> &pte, vector <string> &label, double &cv_error) {
   
   // added for proportional ensemble
   if (CV) cv_error = cross_validation(ptr,10);
   else cv_error = -1;

   assert(ptr.size() > 0);
   vector <DTW_example> tr, te;
   for (int i=0; i<ptr.size(); ++i) {
      tr.push_back(DTW_example(ptr[i]));
   }
   for (int i=0; i<pte.size(); ++i) {
      te.push_back(DTW_example(pte[i]));
   }
   return train_test(tr,te,label);
}
