#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>
#include <cstdlib>

#include "util.h"
#include "example.h"
#include "experiment.h"
#include "cache.h"

using namespace std;

cache C;

bool CV=false; // also output internal cross-validation error

int main(int argc, char **argv) {
   if (argc < 2) {
      cout<<"USAGE: ./main [-cache|-cv] expt_list_file\n";
      exit(1);
   }
   if (string(argv[1])=="-cache") {
      assert(argc == 3);
      compute_caches(argv[2]);
   }
   else {
      string expt_file;
      if (string(argv[1])=="-cv") {
	 CV=true;
	 assert(argc == 3);
	 expt_file = argv[2];
      }
      else expt_file = argv[1];

      ifstream in(expt_file.c_str());
      assert(in);

      string s;
      while (in) {
	 while (s != "<experiments>" && in) in>>s;
	 if (!in) break;
	 vector <classifier> cl;
	 string cachedir;
	 while (s!="</experiments>") {
	    in>>s;
	    if (s=="<cachedir>") {
	       in>>cachedir;
	       in>>s;
	       assert(s=="</cachedir>");
	    }
	    if (s=="<classifiers>") {
	       while (s != "</classifiers>") {
		  in>>s;
		  if (s=="<classifier>") {
		     classifier c;
		     while (s != "</classifier>") {
			in>>s;
			if (s=="<type>") {
			   in>>c.type;
			   in>>s;
			   assert(s=="</type>");
			}
			if (s=="<outdir>") {
			   in>>c.outdir;
			   in>>s;
			   assert(s=="</outdir>");
			}
			if (s=="<kval>") readIntVector(in,s,c.kval);
			if (s=="<kernel>") readStringVector(in,s,c.k);
			if (s=="<n>") readIntVector(in,s,c.n);
			if (s=="<a>") readIntVector(in,s,c.a);
			if (s=="<w>") readIntVector(in,s,c.w);
			if (s=="<r>") readIntVector(in,s,c.r);
			if (s=="<lambda>") readDoubleVector(in,s,c.lambda);
			if (s=="<bin>") readIntVector(in,s,c.bin);
			if (s=="<aggregate>") readIntVector(in,s,c.agg);
			if (s=="<features>") readStringVector(in,s,c.features);
		     }
		     assert(s=="</classifier>");
		     assert(c.type=="Euclidean" || c.type=="DTW" || c.type=="DTW_R" ||c.type=="SAX" ||c.type=="SVM");
		     if (c.type=="SVM") {
			//assert(c.k.size() > 0);
			assert(c.features.size() > 0);
			for (int i=0; i<c.features.size(); ++i) {
			   assert(c.features[i]=="Euclidean" || c.features[i]=="DTW" || c.features[i]=="DTW_R" || c.features[i]=="SAX");
			}
		     }
		     cl.push_back(c);
		  }
	       }
	    }
	    if (s=="<datasets>") {
	       while (s != "</datasets>") {
		  in>>s;
		  if (s=="<dataset>") {
		     string name,trainFile,testFile;
		     vector <int> n,w,a,r;
		     vector <string> k; 
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
			if (s=="<n>") readIntVector(in,s,n);
			if (s=="<a>") readIntVector(in,s,a);
			if (s=="<w>") readIntVector(in,s,w);
			if (s=="<r>") readIntVector(in,s,r);
			if (s=="<kernel>") readStringVector(in,s,k);
		     }
		     experiment ex(name);
		     ex.readExamples(trainFile,testFile);

		     for (int j=0; j<cl.size(); ++j) {
			system(("mkdir "+cl[j].outdir).c_str());
			ofstream out((cl[j].outdir+"/"+ex.giveName()+".txt").c_str());
			cout<<"\n\n"<<ex.giveName()<<"\n";
			out<<ex.giveName()<<"\n";
			if (cl[j].type=="SAX") {
			   vector <int> pn,pw,pa;
			   if (n.size() > 0) pn = n; 
			   else pn = cl[j].n;
			   if (w.size() > 0) pw = w; 
			   else pw = cl[j].w;
			   if (a.size() > 0) pa = a; 
			   else pa = cl[j].a;
			   assert(pn.size() > 0);
			   assert(pw.size() > 0);
			   assert(pa.size() > 0);
			   ex.SAX_train_test(pn,pw,pa,out);
			}
			if (cl[j].type=="Euclidean") {
			   ex.Euclidean_train_test(out);
			}
			if (cl[j].type=="DTW") {
			   readCachedComputationDTW(cachedir,name,ex.trainSize(),ex.testSize());
			   ex.DTW_train_test(out);
			   C.clear();
			}
			if (cl[j].type=="DTW_R") {
			   vector <int> pr;
			   if (r.size() > 0) pr = r; 
			   else pr = cl[j].r;
			   for (int a=0; a<pr.size(); ++a) {
			      readCachedComputationDTW_R(cachedir,name,ex.trainSize(),ex.testSize(),pr[a]);
			   }
			   assert(pr.size() > 0);
			   ex.DTW_R_train_test(pr,out);
			   C.clear();
			}
			if (cl[j].type=="SVM") {
			   vector <int> pn,pw,pa,pr;
			   vector <string> pk;
			   if (n.size() > 0) pn = n; 
			   else pn = cl[j].n;
			   if (w.size() > 0) pw = w; 
			   else pw = cl[j].w;
			   if (a.size() > 0) pa = a; 
			   else pa = cl[j].a;
			   if (r.size() > 0) pr = r; 
			   else pr = cl[j].r;
			   if (k.size() > 0) pk = k; 
			   else pk = cl[j].k;
			   assert(pk.size() > 0);
			   bool fEuclidean=false,fSAX=false,fDTW=false,fDTW_R=false;
			   for (int e=0; e<cl[j].features.size(); ++e) {
			      if (cl[j].features[e]=="Euclidean") fEuclidean=true;
			      if (cl[j].features[e]=="SAX") {
				 fSAX=true;
				 assert(pn.size() > 0);
				 assert(pw.size() > 0);
				 assert(pa.size() > 0);
			      }
			      if (cl[j].features[e]=="DTW") fDTW=true;
			      if (cl[j].features[e]=="DTW_R") {
				 fDTW_R=true;
				 assert(pr.size() > 0);
			      }
			   }
			   if (fDTW) {
			      assert(cachedir != "");
			      readCachedComputationDTW(cachedir,name,ex.trainSize(),ex.testSize());
			   }
			   if (fDTW_R) {
			      assert(cachedir != "");
			      for (int a=0; a<pr.size(); ++a) {
				 readCachedComputationDTW_R(cachedir,name,ex.trainSize(),ex.testSize(),pr[a]);
			      }
			   }
			   ex.SVM_train_test(fEuclidean,fSAX,fDTW,fDTW_R,pn,pw,pa,pr,pk,out);
			   C.clear();
			}
			out<<"\n\n";
			out<<"\n\n";
			out.close();
		     }
		  }
	       }
	    }
	 }
      }
   }
}
