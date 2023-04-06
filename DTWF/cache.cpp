#include "cache.h"
#include "example.h"
#include "DTW_classifier.h"
#include "DTW_R_classifier.h"

extern cache C;

void readExamples_for_caches(vector <example> &e, const string &filename, const string &name, bool ifTrain) {
   // the labels of these examples will not get integerized
   ifstream in(filename.c_str());
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
	 
	 e.push_back(example(v,b[0],e.size(),name,ifTrain));
      }
   }
   in.close();
}

void compute_caches(const string &filename) {
   ifstream in(filename.c_str());
   assert(in);
   string s;
   while (in) {
      while (s != "<experiments>" && in) in>>s;
      if (!in) break;
      string cachedir;
      while (s!="</experiments>") {
	 in>>s;
	 if (s=="<cachedir>") {
	    in>>cachedir;
	    in>>s;
	    assert(s=="</cachedir>");
	    system(("mkdir "+cachedir).c_str());
	 }
	 if (s=="<datasets>") {
	    while (s != "</datasets>") {
	       in>>s;
	       if (s=="<dataset>") {
		  string name, trainFile, testFile;
		  vector <int> r;
		  while (s != "</dataset>") {
		     in>>s;
		     if (s=="<name>") {
			in>>name;
			in>>s;
			assert(s=="</name>");
		     }
		     if (s=="<train>") {
			in>>trainFile;
			in>>s;
			assert(s=="</train>");
		     }
		     if (s=="<test>") {
			in>>testFile;
			in>>s;
			assert(s=="</test>");
		     }
		     if (s=="<r>") readIntVector(in,s,r);
		  }
		  assert(s=="</dataset>");
		  assert(trainFile!="");
		  cout<<"Computing caches for "<<name<<"\n";
		  vector <example> train, test;
		  readExamples_for_caches(train,trainFile,name,true);
		  if (testFile != "") {
		     readExamples_for_caches(test,testFile,name,false);
		  }
      
		  assert(cachedir!="");
		  // compute dtw for training examples
		  vector <DTW_example> dtrain, dtest;
		  for (int i=0; i<train.size(); ++i) {
		     dtrain.push_back(DTW_example(train[i]));
		  }
		  ofstream out((cachedir+"/"+name+"_dtw_tr_tr").c_str());
		  for (int i=0; i<dtrain.size(); ++i) {
		     for (int j=0; j<dtrain.size(); ++j) {
			out<<dtrain[i].dtw(dtrain[j])<<" ";
		     }
		     out<<"\n";
		  }
		  out.close();
		  // compute dtw for test examples
		  if (test.size() > 0) {
		     for (int i=0; i<test.size(); ++i) {
			dtest.push_back(DTW_example(test[i]));
		     }
		     ofstream out((cachedir+"/"+name+"_dtw_tr_te").c_str());
		     for (int i=0; i<dtrain.size(); ++i) {
			for (int j=0; j<dtest.size(); ++j) {
			   out<<dtrain[i].dtw(dtest[j])<<" ";
			}
			out<<"\n";
		     }
		     out.close();
		  }
		  //cout<<r.size()<<"\n";
		  for (int a=0; a<r.size(); ++a) {
		     // compute dtw_r for training examples
		     vector <DTW_R_example> drtrain, drtest;
		     for (int i=0; i<train.size(); ++i) {
			drtrain.push_back(DTW_R_example(train[i]));
		     }
		     ofstream out((cachedir+"/"+name+"_dtw_"+itos(r[a])+"_tr_tr").c_str());
		     for (int i=0; i<drtrain.size(); ++i) {
			for (int j=0; j<drtrain.size(); ++j) {
			   out<<drtrain[i].dtw_r(drtrain[j],r[a])<<" ";
			}
			out<<"\n";
		     }
		     out.close();
		     // compute dtw_r for test examples
		     if (test.size() > 0) {
			for (int i=0; i<test.size(); ++i) {
			   drtest.push_back(DTW_R_example(test[i]));
			}
			ofstream out((cachedir+"/"+name+"_dtw_"+itos(r[a])+"_tr_te").c_str());
			for (int i=0; i<drtrain.size(); ++i) {
			   for (int j=0; j<drtest.size(); ++j) {
			      out<<drtrain[i].dtw_r(drtest[j],r[a])<<" ";
			   }
			   out<<"\n";
			}
			out.close();
		     }
		  }
	       }
	    }
	 }
      }
   }
}

void readCachedComputationDTW(const string &cachedir, const string &name, int trainSize, int testSize) {
   vector <string> sid;
   sid.push_back(name);

   {
   vector <int> iid;
   iid.push_back(0);
   ifstream in((cachedir+"/"+name+"_dtw_tr_tr").c_str());
   if (in) {
      vector <vector <double> > v(trainSize, vector <double> (trainSize));
      for (int i=0; i<trainSize; ++i) {
         for (int j=0; j<trainSize; ++j) {
	    in>>v[i][j];
         }
      }
      in.close();
      C.addCache(sid,iid,v);
   }
   else cout<<"Cache not found.\n";
   }

    
   {
   vector <int> iid;
   iid.push_back(1);
   ifstream in((cachedir+"/"+name+"_dtw_tr_te").c_str());
   if (in) {
      vector <vector <double> > v(trainSize, vector <double> (testSize));
      for (int i=0; i<trainSize; ++i) {
         for (int j=0; j<testSize; ++j) {
	    in>>v[i][j];
         }
      }
      in.close();
      C.addCache(sid,iid,v);
   }
   else cout<<"Cache not found.\n";
   }
}

void readCachedComputationDTW_R(const string &cachedir, const string &name, int trainSize, int testSize, int R) {
   vector <string> sid;
   sid.push_back(name);

   {
   vector <int> iid;
   iid.push_back(0);
   iid.push_back(R);
   ifstream in((cachedir+"/"+name+"_dtw_"+itos(R)+"_tr_tr").c_str());
   if (in) {
      vector <vector <double> > v(trainSize, vector <double> (trainSize));
      for (int i=0; i<trainSize; ++i) {
         for (int j=0; j<trainSize; ++j) {
	    in>>v[i][j];
         }
      }
      in.close();
      C.addCache(sid,iid,v);
   }
   else cout<<"Cache not found.\n";
   }

    
   {
   vector <int> iid;
   iid.push_back(1);
   iid.push_back(R);
   ifstream in((cachedir+"/"+name+"_dtw_"+itos(R)+"_tr_te").c_str());
   if (in) {
      vector <vector <double> > v(trainSize, vector <double> (testSize));
      for (int i=0; i<trainSize; ++i) {
         for (int j=0; j<testSize; ++j) {
	    in>>v[i][j];
         }    
      }
      in.close();
      C.addCache(sid,iid,v);
   }
   else cout<<"Cache not found.\n";
   }
}
