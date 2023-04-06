#include <cassert>
#include <cmath>
#include "util.h"
#include "SAX_classifier.h"

double SAX_example::featureEuclideanDistance(const SAX_example &o) const {
   double r=0;
   for (auto itr=fv.begin(); itr != fv.end(); ++itr) {
      unordered_map <int, double>::const_iterator itr2 = o.fv.find(itr->first);
      if (itr2 != o.fv.end()) {
	 r += pow((itr->second - itr2->second),2);
      }
      else r += pow(itr->second,2);
   }
   for (auto itr2=o.fv.begin(); itr2 != o.fv.end(); ++itr2) {
      if (fv.count(itr2->first)==0)  r += pow(itr2->second,2);
   }
   
   return sqrt(r);
}

double SAX_example::featureDotProduct(const SAX_example &o) const {
   double r=0;
   for (auto itr=fv.begin(); itr != fv.end(); ++itr) {
      unordered_map <int, double>::const_iterator itr2 = o.fv.find(itr->first);
      if (itr2 != o.fv.end()) {
	 r += itr->second*itr2->second;
      }
   }
   return r;
}

double SAX_example::featureCosine(const SAX_example &o) const {
   return featureDotProduct(o)/(sqrt(featureDotProduct(*this)*o.featureDotProduct(o)));
}


void SAX_example::makeSAXfeatures(int n, int w, int a) {
   //normalize(v);
   for (int i=0; i+n<=v.size(); ++i) {
      vector <double> x;
      for (int j=i; j<i+n; ++j) x.push_back(v[j]);
      normalize(x);

      assert(n % w == 0);
      vector <int> word(w,0);
      for (int j=0; j<w; ++j) {
	 double m;
	 for (int k=j*(n/w); k<(j+1)*(n/w); ++k) {
	    m += x[k];
	 }
	 m /= n/w; // average or aggregate

	 word[j] = alphabet(m,a); // alphabatize it
      }

      int nw = baseNum(word,a); // convert the word into an integer for hashing
      unordered_map <int, double>::iterator itr = fv.find(nw);
      if (itr==fv.end()) {
	 fv[nw] = 1;
      }
      else {
	 fv[nw] = fv[nw] + 1; 
      }
   }
}

/*double SAX_classifier::train_test(const vector <SAX_example> &ptr, const vector <SAX_example> &pte, vector <string> &label) {
   assert(ptr.size() > 0);

   label.resize(pte.size());
   vector <string> corr_label;
   // one nearest neighbor classification using cosines
   for (int i=0; i<pte.size(); ++i) {
      corr_label.push_back(pte[i].giveLabel());
      double maxD=ptr[0].featureCosine(pte[i]);
      int c=0;
      for (int j=1; j<ptr.size(); ++j) {
	 double d=ptr[j].featureCosine(pte[i]);
	 if (d > maxD) {
	    maxD = d;
	    c=j;
	 }
      }
      label[i] = ptr[c].giveLabel();
   }
   return compute_error(corr_label,label);
}
*/

double SAX_classifier::train_test(const vector <SAX_example> &ptr, const vector <SAX_example> &pte, vector <string> &label) {
   assert(ptr.size() > 0);

   label.resize(pte.size());
   vector <string> corr_label;
   // one nearest neighbor classification using Euclidean distance
   for (int i=0; i<pte.size(); ++i) {
      corr_label.push_back(pte[i].giveLabel());
      double minD=ptr[0].featureEuclideanDistance(pte[i]);
      int c=0;
      for (int j=1; j<ptr.size(); ++j) {
	 double d=ptr[j].featureEuclideanDistance(pte[i]);
	 if (d < minD) {
	    minD = d;
	    c=j;
	 }
      }
      label[i] = ptr[c].giveLabel();
   }
   return compute_error(corr_label,label);
}

double SAX_classifier::cross_validation(const vector <example> &ptr, int pn, int pw, int pa, int folds) {
   assert(ptr.size() > 0);
      
   vector <SAX_example> tr;
   for (int i=0; i<ptr.size(); ++i) {
      tr.push_back(SAX_example(ptr[i],pn,pw,pa));
   }

   vector <string> outLabel, corrLabel;
   for (int f=0; f<folds; ++f) {
      // make train-test examples for the fold
      vector <SAX_example> ftr,fte;
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

double SAX_classifier::train_test(const vector <example> &ptr, const vector <example> &pte, const vector <int> &pn, const vector <int> &pw, const vector <int> &pa) {
   int bn,bw,ba;
   vector <string> label;
   return train_test(ptr,pte,pn,pw,pa,bn,bw,ba,label);
}

double SAX_classifier::train_test(const vector <example> &ptr, const vector <example> &pte, const vector <int> &pn, const vector <int> &pw, const vector <int> &pa, int &bn, int &bw, int &ba, vector <string> &label) {
   assert(ptr.size() > 0);
   vector <SAX_example> tr, te;
   if (pn.size()==1 && pw.size()==1 && pa.size()==1) {
      bn = pn[0];
      bw = pw[0];
      ba = pa[0];
   }
   else {
      assert(pn.size()>0 && pw.size()>0 && pa.size()>0);
      // find the best parameters by cross-validation within the training data
      int FOLDS=5;
      double minErr=999;

      for (int i=0; i<pn.size(); ++i) {
	 for (int j=0; j<pw.size(); ++j) {
	    for (int k=0; k<pa.size(); ++k) {
	       double err = cross_validation(ptr,pn[i],pw[j],pa[k],FOLDS);
	       if (err < minErr) {
		  minErr=err;
		  bn = pn[i];
		  bw = pw[j];
		  ba = pa[k];
	       }
	    }
	 }
      }
   }
   for (int i=0; i<ptr.size(); ++i) {
      tr.push_back(SAX_example(ptr[i],bn,bw,ba));
   }
   for (int i=0; i<pte.size(); ++i) {
      te.push_back(SAX_example(pte[i],bn,bw,ba));
   }
   return train_test(tr,te,label);
}
