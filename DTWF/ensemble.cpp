#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <glob.h>

using namespace std;

inline std::vector<std::string> glob(const std::string& pat){
   // glob function 
   glob_t glob_result;
   glob(pat.c_str(),GLOB_TILDE,NULL,&glob_result);
   vector<string> ret;
   for(unsigned int i=0;i<glob_result.gl_pathc;++i){
      ret.push_back(string(glob_result.gl_pathv[i]));
   }
   globfree(&glob_result);
   return ret;
}


class method_result { // result of a method
   public:
   vector <string> corr, label; // correct and output labels
   double cv_error; // cross-validation error

   void read(const string &filename) {
      ifstream in(filename.c_str());
      assert(in);
      string s;
      while (s != "CV_Error:") in>>s;
      if (!in) {
	 cout<<"CV_Error: not found in "<<filename<<"\n";
	 exit(1);
      }
      in>>cv_error;

      while (s != "<correct_and_label>") in>>s;
      if (!in) {
	 cout<<"<correct_and_label> not found in "<<filename<<"\n";
	 exit(1);
      }

      while (s != "</correct_and_label>") {
	 in>>s;
	 if (!in) {
	    cout<<"</correct_and_label> not found in "<<filename<<"\n";
	    exit(1);
	 }
	 if (s != "</correct_and_label>") {
	    corr.push_back(s);
	    in>>s;
	    label.push_back(s);
	 }
      }
      in.close();
   }

   double compute_error() const {
      assert(corr.size()==label.size());
      assert(corr.size() > 0);

      double correct=0;
      for (int i=0; i<corr.size(); ++i) {
	 if (corr[i]==label[i]) ++correct;
      }

      return 1-(correct/corr.size());
   }

   void write(const string &filename) const{
      ofstream out(filename.c_str());
      assert(out);
      out<<"Error: "<<compute_error()<<"\n";
      out<<"<correct_and_label>\n";
      assert(corr.size()==label.size());
      for (int i=0; i<corr.size(); ++i) {
	 out<<corr[i]<<" "<<label[i]<<"\n";
      }
      out<<"</correct_and_label>\n";
      out.close();
   }
};

method_result ensemble(const vector <method_result> &mr, const string &ensemble_type) {
   assert(mr.size() > 0);
   assert(ensemble_type=="PROP" || ensemble_type=="EQUAL" || ensemble_type=="BEST");

   for (int i=1; i<mr.size(); ++i) {
      assert(mr[i].corr==mr[0].corr);
      assert(mr[i].label.size()==mr[i].label.size());
   }

   vector <double> w(mr.size()); // weights 
   if (ensemble_type=="PROP") {
      double total=0; // sum of accuracies
      for (int i=0; i<mr.size(); ++i) total += 1-mr[i].cv_error;
      for (int i=0; i<w.size(); ++i) w[i] = (1-mr[i].cv_error)/total;
   }
   if (ensemble_type=="EQUAL") {
      for (int i=0; i<w.size(); ++i) w[i] = 1.0/mr.size();
   }
   if (ensemble_type=="BEST") {
      int c=0; // index of the best one
      for (int i=1; i<mr.size(); ++i) {
	 if (mr[i].cv_error < mr[c].cv_error) c = i;
      }
      for (int i=0; i<w.size(); ++i) w[i] = 0;
      w[c] = 1;
   }
   cout<<"Cross-validation weights:\n";
   for (int i=0; i<w.size(); ++i) {
      cout<<w[i]<<" ";
   }
   cout<<"\n\n";

   method_result e;
   e.corr = mr[0].corr;

   for (int k=0; k<mr[0].label.size(); ++k) { 
      // assign label to ensemble for each example

      // collect labels and their weights
      vector <string> label;
      vector <double> wlabel;
      for (int i=0; i<mr.size(); ++i) {
	 bool f=false;
	 for (int j=0; j<label.size(); ++j) {
	    if (label[j]==mr[i].label[k]) {
	       wlabel[j] += w[i];
	       f=true;
	       break;
	    }
	 }
	 if (!f) {
	    label.push_back(mr[i].label[k]);
	    wlabel.push_back(w[i]);
	 }
      }

      // assign the most weighted label
      int c=0;
      for (int j=1; j<label.size(); ++j) {
	 if (wlabel[j] > wlabel[c]) c=j;
      }
      e.label.push_back(label[c]);

      //for (int j=0; j<label.size(); ++j) {
	// cout<<label[j]<<":"<<wlabel[j]<<" ";
      //}
      //cout<<" -> "<<label[c]<<"\n";
   }
   return e;
}

string removeDir(const string &s) {
   // removes dir name from the front
   string r;
   for (int i=(int)s.size()-1; i>=0; --i) {
      if (s[i] != '/') r.insert(r.begin(), s[i]);
      else break;
   }
   return r;
}

int main(int argc, char **argv) {
   if (argc < 3) {
      cout<<"USAGE: ./ensemble outdir indir1 [indir2]*\n";
      exit(1);
   }

   string outdir = argv[1];
   system(("mkdir "+outdir).c_str());
   vector <string> indir;
   for (int i=2; i<argc; ++i) {
      indir.push_back(argv[i]);
   }

   vector <string> dataname = glob((indir[0]+"/*.txt").c_str());
   for (int i=0; i<dataname.size(); ++i) {
      dataname[i] = removeDir(dataname[i]);
   }

   for (int i=0; i<dataname.size(); ++i) { // process for each dataset
      cout<<dataname[i]<<"\n";
      vector <method_result> m(indir.size());
      for (int j=0; j<indir.size(); ++j) { // read results from all methods
	 m[j].read(indir[j]+"/"+dataname[i]);
      }
      //method_result em = ensemble(m,"EQUAL"); // create ensemble result
      //method_result em = ensemble(m,"BEST"); // create ensemble result
      method_result em = ensemble(m,"PROP"); // create ensemble result
      em.write(outdir+"/"+dataname[i]); // write ensemble result
   }
}
