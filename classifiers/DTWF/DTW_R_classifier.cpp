#include <cassert>
#include <cmath>
#include <cfloat>
#include "util.h"
#include "DTW_R_classifier.h"
#include "cache.h"

extern cache C;
extern bool CV;

double DTW_R_example::dtw_r(const DTW_R_example &o, int R) const {
   // computes DTW_R distance, first checks cache othewise computes
   double r;
   vector <int> iid;
   if (ifTrain) {
      if (o.ifTrain) iid.push_back(0);
      else iid.push_back(1);
      iid.push_back(R);

      double r;
      if (C.giveValue(vector <string>(1,name),iid,id,o.id,r)) {
	 return r;
      }
   }
 
   // not in cache
   return compute_dtw_r(o,R);
}

double DTW_R_example::compute_dtw_r(const DTW_R_example &o, int R) const {
   vector <vector <double> > d(v.size()+1, vector <double> (o.v.size()+1,DBL_MAX));
   
   int w = max(ceil((v.size()*R)/100.0),double(abs((int)v.size()-(int)o.v.size()))); // R is percent
   //int w = max(R,abs((int)v.size()-(int)o.v.size())); // R is absolute
   d[0][0] = 0;
   for (int i=1; i<=v.size(); ++i) {
      for (int j=max(1,i-w); j<=min(i+w,(int)o.v.size()); ++j) {
	 double mi=d[i-1][j];
	 if (mi > d[i][j-1]) mi=d[i][j-1];
	 if (mi > d[i-1][j-1]) mi=d[i-1][j-1];
	 d[i][j] = pow((v[i-1]-o.v[j-1]),2) + mi;
      }
   }
   return sqrt(d[v.size()][o.v.size()]);
}

double DTW_R_classifier::train_test(const vector <DTW_R_example> &pptr, const vector <DTW_R_example> &pte, int R, vector <string> &label) {
   assert(pptr.size() > 0);

   vector <DTW_R_example> ptr=pptr;

   //ptr.resize((pptr.size()*100)/100); // training percent
   //cout<<ptr.size()<<"\n";


   label.resize(pte.size());
   vector <string> corr_label;
   // one nearest neighbor classification using DTW_R 
   for (int i=0; i<pte.size(); ++i) {
      corr_label.push_back(pte[i].giveLabel());
      double minD=ptr[0].dtw_r(pte[i],R);
      int c=0;
      for (int j=1; j<ptr.size(); ++j) {
	 double d=ptr[j].dtw_r(pte[i],R);
	 if (d < minD) {
	    minD = d;
	    c=j;
	 }
      }
      label[i] = ptr[c].giveLabel();
   }
   return compute_error(corr_label,label);
}

double DTW_R_classifier::cross_validation(const vector <example> &ptr, int R, int folds) {
   assert(ptr.size() > 0);
      
   vector <DTW_R_example> tr;
   for (int i=0; i<ptr.size(); ++i) {
      tr.push_back(DTW_R_example(ptr[i]));
   }

   vector <string> outLabel, corrLabel;
   for (int f=0; f<folds; ++f) {
      // make train-test examples for the fold
      vector <DTW_R_example> ftr,fte;
      for (int i=0; i<tr.size(); ++i) {
	 if (i % folds == f) {
	    fte.push_back(tr[i]);
	    corrLabel.push_back(tr[i].giveLabel());
	 }
	 else ftr.push_back(tr[i]);
      }
      vector <string> label;
      train_test(ftr,fte,R,label);
      for (int i=0; i<label.size(); ++i) outLabel.push_back(label[i]);
   }

   return compute_error(corrLabel,outLabel);
}

double DTW_R_classifier::train_test(const vector <example> &ptr, const vector <example> &pte, const vector <int> &pR) {
   int bR;
   vector <string> label;
   double cv_error;
   return train_test(ptr,pte,pR,bR,label,cv_error);
}

double DTW_R_classifier::train_test(const vector <example> &ptr, const vector <example> &pte, const vector <int> &pR, int &bR, vector <string> &label, double &cv_error) {
   assert(ptr.size() > 0);
   vector <DTW_R_example> tr, te;

   if (pR.size()==1) {
      bR = pR[0];

      // added for proportional ensemble
      if (CV) cv_error = cross_validation(ptr,bR,10);
      else cv_error = -1;
   }
   else { 
      assert(pR.size() > 0);
       // find the best parameter by cross-validation within the training data
      int FOLDS=10;
      double minErr=999;

      for (int i=0; i<pR.size(); ++i) {
	 double err = cross_validation(ptr,pR[i],FOLDS);
	 cout<<"R="<<pR[i]<<" Error="<<err<<"\n";
	 if (err < minErr) {
	    minErr = err;
	    bR = pR[i];
	 }
      }
      cv_error = minErr;
   }

   for (int i=0; i<ptr.size(); ++i) {
      tr.push_back(DTW_R_example(ptr[i]));
   }
   for (int i=0; i<pte.size(); ++i) {
      te.push_back(DTW_R_example(pte[i]));
   }
   return train_test(tr,te,bR,label);
}
