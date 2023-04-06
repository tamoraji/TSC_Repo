#include <cstdlib>
#include <cassert>
#include <algorithm>
#include "util.h"
#include "Euclidean_classifier.h"
#include "SAX_classifier.h"
#include "DTW_classifier.h"
#include "DTW_R_classifier.h"
#include "SVM_classifier.h"
#include "experiment.h"

void experiment::readExamples(const string &trainFile, const string &testFile) {
   ifstream in(trainFile.c_str());
   cout<<trainFile<<"\n";
   assert(in);
   while (in) {
      string s;
      getline(in,s);
      if (in) {
	 vector <string> b;
	 breakSpaces(s,b);
	 assert(b.size() > 1);
	 vector <double> v;
	 for (int i=1; i<b.size(); ++i) v.push_back(atof(b[i].c_str()));
	 
	 // get label name of type integer from 0 onward
	 string bl;
	 for (int i=0; i<labelMap.size(); ++i) {
	    if (b[0]==labelMap[i]) { 
	       bl = itos(i);
	       break;
	    }
	 }
	 if (bl=="") {
	    labelMap.push_back(b[0]);
	    bl = itos(labelMap.size()-1);
	 }

	 train.push_back(example(v,bl,train.size(),name,true));
      }
   }
   in.close();

   in.open(testFile.c_str());
   cout<<testFile<<"\n";
   assert(in);
   while (in) {
      string s;
      getline(in,s);
      if (in) {
	 vector <string> b;
	 breakSpaces(s,b);
	 assert(b.size() > 1);
	 vector <double> v;
	 for (int i=1; i<b.size(); ++i) v.push_back(atof(b[i].c_str()));

	 // get label name of type integer from 0 onward
	 string bl;
	 for (int i=0; i<labelMap.size(); ++i) {
	    if (b[0]==labelMap[i]) { 
	       bl = itos(i);
	       break;
	    }
	 }
	 if (bl=="") {
	    labelMap.push_back(b[0]);
	    bl = itos(labelMap.size()-1);
	 }

	 test.push_back(example(v,bl,test.size(),name,false));
      }
   }
   in.close();
}

double experiment::Euclidean_train_test(ostream &out) const {
   Euclidean_classifier sc;
   vector <string> label;
   double error = sc.train_test(train,test,label);
   out<<"Error: "<<error<<"\n";
   return error;
}

double experiment::SAX_train_test(const vector <int> &n, const vector <int> &w, const vector <int> &a, ostream &out) const {
   SAX_classifier sc;
   int bn, bw, ba;
   vector <string> label;
   double error = sc.train_test(train,test,n,w,a,bn,bw,ba,label);
   out<<"Error: "<<error<<"\n";
   out<<"For: n="<<bn<<" w="<<bw<<" a="<<ba<<"\n";
   return error;
}

double experiment::DTW_train_test(ostream &out) const {
   DTW_classifier sc;
   vector <string> label;
   double cv_error=-1;
   double error = sc.train_test(train,test,label,cv_error);
   out<<"Error: "<<error<<"\n";
   out<<"CV_Error: "<<cv_error<<"\n";
   out<<"<correct_and_label>\n";
   assert(test.size()==label.size());
   for (int i=0; i<test.size(); ++i) {
      out<<labelName(test[i].giveLabel())<<" "<<labelName(label[i])<<"\n";
   }
   out<<"</correct_and_label>\n";
   return error;
}


double experiment::DTW_R_train_test(const vector <int> &r, ostream &out) const {
   DTW_R_classifier sc;
   int br;
   vector <string> label;
   double cv_error=-1;
   double error = sc.train_test(train,test,r,br,label,cv_error);
   out<<"Error: "<<error<<"\n";
   out<<"For: r="<<br<<"\n";
   out<<"CV_Error: "<<cv_error<<"\n";
   out<<"<correct_and_label>\n";
   assert(test.size()==label.size());
   for (int i=0; i<test.size(); ++i) {
      out<<labelName(test[i].giveLabel())<<" "<<labelName(label[i])<<"\n";
   }
   out<<"</correct_and_label>\n";
   return error;
}

double experiment::SVM_train_test(bool fEuclidean, bool fSAX, bool fDTW, bool fDTW_R, const vector <int> &n, const vector <int> &w, const vector <int> &a, const vector <int> &r, const vector <string> &k, ostream &out) const {
   SVM_classifier sc;
   int bn, bw, ba, br;
   string bk;
   vector <string> label;
   double cv_error=-1;
   double error = sc.train_test(train,test,fEuclidean,fSAX,fDTW,fDTW_R,n,w,a,r,k,bn,bw,ba,br,bk,label,cv_error);
   out<<"Error: "<<error<<"\nFor:\nk="<<bk<<"\n";
   if (fSAX) {
      out<<"n="<<bn<<" w="<<bw<<" a="<<ba<<"\n";
   }
   if (fDTW_R) {
      out<<"r="<<br<<"\n";
   }
   out<<"CV_Error: "<<cv_error<<"\n";
   out<<"\n\n";

   out<<"<correct_and_label>\n";
   assert(test.size()==label.size());
   for (int i=0; i<test.size(); ++i) {
      out<<labelName(test[i].giveLabel())<<" "<<labelName(label[i])<<"\n";
   }
   out<<"</correct_and_label>\n";
   return error;
}
