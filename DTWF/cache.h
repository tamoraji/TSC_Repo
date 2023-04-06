#ifndef _CACHE_H
#define _CACHE_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

class cachedComputation { // use to store DTW distances
   public:
   vector <string> sid; // name (of dataset)
   vector <int> iid; // 0/1 (train-train/train-test), R (of DTW_R)

   vector <vector <double> > v;  // distances

   cachedComputation(const vector <string> &psid, const vector <int> &piid, const  vector <vector <double> > &pv) {
      sid = psid;
      iid = piid;
      v = pv;
   }
};

class cache {
   vector <cachedComputation> c;
   public:

   int size() const {
      return c.size();
   }

   void clear() {
      c.clear();
   }

   void addCache(const vector <string> &psid, const vector <int> &piid, const  vector <vector <double> > &pv) {
      for (int i=0; i<c.size(); ++i) {
	 if (piid==c[i].iid && psid==c[i].sid) { // overwrite
	    c[i].v = pv;
	    return;
	 }
      }
      c.push_back(cachedComputation(psid,piid,pv));
   }

   bool giveValue(const vector <string> &ss, const vector <int> &ii, int a, int b, double &value) const {
      for (int i=0; i<c.size(); ++i) { 
	 if (ii==c[i].iid && ss==c[i].sid) {
	    if (a < c[i].v.size()) {
	       if (b < c[i].v[a].size()) {
		  value = c[i].v[a][b];
		  return true;
	       }
	    }
	    break;
	 }
      }
      return false;
   }
};

void compute_caches(const string &filename);

void readCachedComputationDTW(const string &cachedir, const string &name, int trainSize, int testSize);
void readCachedComputationDTW_R(const string &cachedir, const string &name, int trainSize, int testSize, int R);

#endif
