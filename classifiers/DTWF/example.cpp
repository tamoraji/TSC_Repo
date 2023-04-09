#include <cassert>
#include <cmath>
#include "example.h"
#include "util.h"

double compute_error(const vector <example> &e, const vector <string> &out_label) {
   assert(e.size() > 0);
   assert(e.size() == out_label.size());

   int corr=0;
   for (int i=0; i<e.size(); ++i) {
      if (e[i].giveLabel()==out_label[i]) ++corr;
   }

   return  1-((double)corr/(double)e.size()); 
}

double compute_error(const vector <string> &corr_label, const vector <string> &out_label) {
   assert(corr_label.size() > 0);
   assert(corr_label.size() == out_label.size());

   int corr=0;
   for (int i=0; i<corr_label.size(); ++i) {
      if (corr_label[i]==out_label[i]) ++corr;
   }

   return  1-((double)corr/(double)corr_label.size()); 
}
