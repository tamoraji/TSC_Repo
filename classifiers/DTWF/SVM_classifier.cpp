#include <cassert>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include "util.h"
#include "SVM_classifier.h"
#include "cache.h"

string LIBSVMDIR = "/data02/code/katerj/timeseries/code/libsvm-3.17";
extern cache C;
extern bool CV;

double SVM_example::kernelUn(const SVM_example &o) const { // unnormalized kernel
   double r=0;

   if (fEuclidean) {
      r += dotProduct(feature_Euclidean, o.feature_Euclidean);
   }
   if (fDTW) {
      r += dotProduct(feature_DTW, o.feature_DTW);
   }
   if (fDTW_R) {
      r += dotProduct(feature_DTW_R, o.feature_DTW_R);
   }
   if (fSAX) {
      for (auto itr=fv.begin(); itr != fv.end(); ++itr) {
          unordered_map <int, double>::const_iterator itr2 = o.fv.find(itr->first);
          if (itr2 != o.fv.end()) {
	    r += itr->second*itr2->second;
         }
      }
   }
   assert(ktype==o.ktype);
   if (ktype=="linear") return (1+r); // linear
   if (ktype=="poly2") return (1+r)*(1+r); // polynomial 2
   if (ktype=="poly3") return (1+r)*(1+r)*(1+r); // polynomial 3
   assert(false);
}

double SVM_example::euclidean_distance(const SVM_example &o) const {
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


double SVM_example::dtw(const SVM_example &o) const {
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

double SVM_example::compute_dtw(const SVM_example &o) const {
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

void SVM_example::addEuclideanfeatures(const vector <SVM_example> &tr) {
   fEuclidean=true;
   for (int i=0; i<tr.size(); ++i) {
      feature_Euclidean.push_back(tr[i].euclidean_distance(*this));
   }
}

void SVM_example::addDTWfeatures(const vector <SVM_example> &tr) {
   fDTW=true;
   for (int i=0; i<tr.size(); ++i) {
      feature_DTW.push_back(tr[i].dtw(*this));
   }
}

double SVM_example::dtw_r(const SVM_example &o, int R) const {
   // dtw_r distance, first checks the cache otherwise computes
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

double SVM_example::compute_dtw_r(const SVM_example &o, int R) const {
   vector <vector <double> > d(v.size()+1, vector <double> (o.v.size()+1,DBL_MAX));
   
   int w = ceil((v.size()*R)/100.0);
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

void SVM_example::addDTW_Rfeatures(const vector <SVM_example> &tr, int R) {
   fDTW_R=true;
   for (int i=0; i<tr.size(); ++i) {
      feature_DTW_R.push_back(tr[i].dtw_r(*this,R));
   }
}

void SVM_example::addSAXfeatures(int n, int w, int a) {
   fSAX=true;
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

      int nw = baseNum(word,a);
      unordered_map <int, double>::iterator itr = fv.find(nw);
      if (itr==fv.end()) {
	 fv[nw] = 1;
      }
      else {
	 fv[nw] = fv[nw] + 1; 
      }
   }
}

void SVM_classifier::svm_train(const vector <SVM_example> &tr) {
   // uses libsvm to train
   assert(tr.size() > 0);

   ofstream out("tmp_train_file");
   assert(out);
   for (int i=0; i<tr.size(); ++i) {
      out<<tr[i].giveLabel()<<" 0:"<<i+1;
      for (int j=0; j<tr.size(); ++j) {
         out<<" "<<j+1<<":"<<tr[i].kernelUn(tr[j]);
      }
      out<<"\n";
   }
   out.close();
   string command = LIBSVMDIR + "/svm-train -q -t 4 -c 1 tmp_train_file tmp_model_file";
   system(command.c_str());
}

void SVM_classifier::svm_test(const vector <SVM_example> &tr, const vector <SVM_example> &te, vector <string> &label) {
   // uses libsvm to test
   label.resize(te.size());
   ofstream out("tmp_test_file");
   assert(out);

   for (int i=0; i<te.size(); ++i) {
      out<<te[i].giveLabel()<<" 0:"<<i+1;
      for (int j=0; j<tr.size(); ++j) {
         out<<" "<<j+1<<":"<<te[i].kernelUn(tr[j]);
      }
      out<<"\n";
   }
   out.close();
   string command = LIBSVMDIR + "/svm-predict tmp_test_file  tmp_model_file tmp_out_file";

   system(command.c_str());

   ifstream in("tmp_out_file");
   assert(in);
   string s;
   for (int i=0; i<te.size(); ++i) {
      in>>label[i];
   }
   in.close();
}

double SVM_classifier::train_test(const vector <SVM_example> &ptr, const vector <SVM_example> &pte, vector <string> &label) {
   vector <SVM_example> tr=ptr;

   //tr.resize((tr.size()*100)/100); // training percent
   //cout<<tr.size()<<"\n";

   svm_train(tr);
   vector <string> corr_label;
   for (int i=0; i<pte.size(); ++i) {
      corr_label.push_back(pte[i].giveLabel());
   }
   svm_test(tr,pte,label);
   return compute_error(corr_label,label);
}

double SVM_classifier::cross_validation(const vector <example> &ptr, bool pfEuclidean, bool pfSAX, bool pfDTW, bool pfDTW_R, int pn, int pw, int pa, int R, const string &k, int folds) {
   assert(ptr.size() > 0);

   vector <string> outLabel, corrLabel;
   for (int f=0; f<folds; ++f) {
      // make train-test examples for the fold
      vector <SVM_example> ftr,fte;
      for (int i=0; i<ptr.size(); ++i) {
	 if (i % folds == f) {
	    fte.push_back(SVM_example(ptr[i]));
	    corrLabel.push_back(ptr[i].giveLabel());
	 }
	 else ftr.push_back(SVM_example(ptr[i]));
      }
      for (int i=0; i<ftr.size(); ++i) {
	 ftr[i].setK(k);
	 if (pfEuclidean) {
	    ftr[i].addEuclideanfeatures(ftr);
	 }
	 if (pfSAX) {
	    ftr[i].addSAXfeatures(pn,pw,pa);
	 }
	 if (pfDTW) {
	    ftr[i].addDTWfeatures(ftr);
	 }
	 if (pfDTW_R) {
	    ftr[i].addDTW_Rfeatures(ftr,R);
	 }
      }
      for (int i=0; i<fte.size(); ++i) {
	 fte[i].setK(k);
	 if (pfEuclidean) {
	    fte[i].addEuclideanfeatures(ftr);
	 }
	 if (pfSAX) {
	    fte[i].addSAXfeatures(pn,pw,pa);
	 }
	 if (pfDTW) {
	    fte[i].addDTWfeatures(ftr);
	 }
	 if (pfDTW_R) {
	    fte[i].addDTW_Rfeatures(ftr,R);
	 }
      }
      vector <string> label;
      train_test(ftr,fte,label);
      for (int i=0; i<label.size(); ++i) outLabel.push_back(label[i]);
   }

   return compute_error(corrLabel,outLabel);
}

double SVM_classifier::train_test(const vector <example> &ptr, const vector <example> &pte, bool pfEuclidean, bool pfSAX, bool pfDTW, bool pfDTW_R, const vector <int> &pn, const vector <int> &pw, const vector <int> &pa, const vector <int> &pR, const vector <string> &pk) {
   int bn,bw,ba,bR;
   string bk;
   vector <string> label;
   double cv_error;
   return train_test(ptr,pte,pfEuclidean,pfSAX,pfDTW,pfDTW_R,pn,pw,pa,pR,pk,bn,bw,ba,bR,bk,label,cv_error);
}

double SVM_classifier::train_test(const vector <example> &ptr, const vector <example> &pte, bool pfEuclidean, bool pfSAX, bool pfDTW,  bool pfDTW_R, const vector <int> &pn, const vector <int> &pw, const vector <int> &pa, const vector <int> &pR, const vector <string> &pk, int &bn, int &bw, int &ba, int &bR, string &bk, vector <string> &label, double &cv_error) {
   assert(ptr.size() > 0);
   bool cv=false; // need for cross-validation
   if (pk.size() > 1) {
      cv = true;
   }
   else {
      bk=pk[0];
   }
   if (pfSAX) {
      if (pn.size() > 1 || pw.size() > 1 || pa.size() > 1) cv=true; 
      else {
	 bn = pn[0];
	 bw = pw[0];
	 ba = pa[0];
      }
   }
   if (pfDTW_R) {
      if (pR.size() > 1) cv=true;
      else {
	 bR = pR[0];
      }
   }
   if (cv) {
      // find the best parameters by cross-validation within the training data
      int FOLDS=10;
      double minErr=999;

      if (!pfSAX && !pfDTW_R) {
	 for (int n=0; n<pk.size(); ++n) {
	    double err = cross_validation(ptr,pfEuclidean,pfSAX,pfDTW,pfDTW_R,-1,-1,-1,-1,pk[n],FOLDS);
	    if (err < minErr) {
	       minErr=err;
	       bk = pk[n];
	    }
	 }
      }
      if (!pfSAX && pfDTW_R) {
	 for (int m=0; m<pR.size(); ++m) {
	    for (int n=0; n<pk.size(); ++n) {
	       double err = cross_validation(ptr,pfEuclidean,pfSAX,pfDTW,pfDTW_R,-1,-1,-1,pR[m],pk[n],FOLDS);
	       if (err < minErr) {
		  minErr=err;
		  bR = pR[m];
		  bk = pk[n];
	       }
	    }
	 }
      }
      if (pfSAX && !pfDTW_R) {
	 for (int i=0; i<pn.size(); ++i) {
	    for (int j=0; j<pw.size(); ++j) {
	       for (int l=0; l<pa.size(); ++l) {
		  for (int n=0; n<pk.size(); ++n) {
		     double err = cross_validation(ptr,pfEuclidean,pfSAX,pfDTW,pfDTW_R,pn[i],pw[j],pa[l],-1,pk[n],FOLDS);
		     if (err < minErr) {
			minErr=err;
			bn = pn[i];
			bw = pw[j];
			ba = pa[l];
			bk = pk[n];
		     }
		  }
	       }
	    }
	 }
      }
      if (pfSAX && pfDTW_R) {
	 for (int i=0; i<pn.size(); ++i) {
	    for (int j=0; j<pw.size(); ++j) {
	       for (int l=0; l<pa.size(); ++l) {
		  for (int m=0; m<pR.size(); ++m) {
		     for (int n=0; n<pk.size(); ++n) {
			double err = cross_validation(ptr,pfEuclidean,pfSAX,pfDTW,pfDTW_R,pn[i],pw[j],pa[l],pR[m],pk[n],FOLDS);
			if (err < minErr) {
			   minErr=err;
			   bn = pn[i];
			   bw = pw[j];
			   ba = pa[l];
			   bR = pR[m];
			   bk = pk[n];
			}
		     }
		  }
	       }
	    }
	 }
      }
      cv_error = minErr; 
   }

   if (!cv && CV) { // added for proportional ensemble
      cv_error = cross_validation(ptr,pfEuclidean,pfSAX,pfDTW,pfDTW_R,bn,bw,ba,bR,bk,10);
   }
   else {
      cv_error = -1;
   }


   vector <SVM_example> tr, te;
   for (int i=0; i<ptr.size(); ++i) {
      tr.push_back(SVM_example(ptr[i]));
   }
   for (int i=0; i<tr.size(); ++i) {
      tr[i].setK(bk);
      if (pfEuclidean) {
	 tr[i].addEuclideanfeatures(tr);
      }
      if (pfSAX) {
	 tr[i].addSAXfeatures(bn,bw,ba);
      }
      if (pfDTW) {
	 tr[i].addDTWfeatures(tr);
      }
      if (pfDTW_R) {
	 tr[i].addDTW_Rfeatures(tr,bR);
      }
   }
   for (int i=0; i<pte.size(); ++i) {
      te.push_back(SVM_example(pte[i]));
   }
   for (int i=0; i<te.size(); ++i) {
      te[i].setK(bk);
      if (pfEuclidean) {
	 te[i].addEuclideanfeatures(tr);
      }
      if (pfSAX) {
	 te[i].addSAXfeatures(bn,bw,ba);
      }
      if (pfDTW) {
	 te[i].addDTWfeatures(tr);
      }
      if (pfDTW_R) {
	 te[i].addDTW_Rfeatures(tr,bR);
      }
   }
   return train_test(tr,te,label);
}
