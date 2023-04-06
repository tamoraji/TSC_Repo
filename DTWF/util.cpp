#include <cassert>
#include <cmath>
#include <cstdlib>
#include "util.h"

double mean(const vector <double> &v) {
   assert(v.size()>0);
   double sum = 0;
   for (int i=0; i<v.size(); ++i) sum += v[i];
   return sum/v.size();
}

double stdDev(const vector <double> &v) {
   assert(v.size()>0);
   double sum = 0;
   double m = mean(v);
   for (int i=0; i<v.size(); ++i) sum += pow((v[i]-m),2);
   return sqrt(sum/v.size());
}

void normalize(vector <double> &v) {
   double m=mean(v);
   double sd=stdDev(v);
   if (sd!=0) {
      for (int i=0; i<v.size(); ++i) {
	 v[i] = (v[i]-m)/sd;
      }
   }
   else { // zero standard deviation
      for (int i=0; i<v.size(); ++i) {
	 v[i] = (v[i]-m);
      }
   }

}

int baseNum(const vector <int> &d, int a) {
   assert(d.size() > 0);
   int r=0;
   for (int i=0; i<d.size()-1; ++i) r = (r+d[i])*a;
   r += d[d.size()-1];
   return r;
}

double dotProduct(const vector <double> &v1, const vector <double> &v2) {
   assert(v1.size()==v2.size());
   double r=0;
   for (int i=0; i<v1.size(); ++i) {
      r += v1[i]*v2[i];
   }
   return r;
}

double alphabet(double m, int a) {
   assert(a >=2 && a <=10);
   if (a==2) {
      if (m < 0) return 0;
      return 1;
   }
   if (a==3) {
      if (m < -0.43) return 0;
      if (m < 0.43) return 1;
      return 2;
   }
   if (a==4) {
      if (m < -0.67) return 0;
      if (m < 0) return 1;
      if (m < 0.67) return 2;
      return 3;
   }
   if (a==5) {
      if (m < -0.84) return 0;
      if (m < -0.25) return 1;
      if (m < 0.25) return 2;
      if (m < 0.84) return 3;
      return 4;
   }
   if (a==6) {
      if (m < -0.97) return 0;
      if (m < -0.43) return 1;
      if (m < 0) return 2;
      if (m < 0.43) return 3;
      if (m < 0.97) return 4;
      return 5;
   }
   if (a==7) {
      if (m < -1.07) return 0;
      if (m < -0.57) return 1;
      if (m < -0.18) return 2;
      if (m < 0.18) return 3;
      if (m < 0.57) return 4;
      if (m < 1.07) return 5;
      return 6;
   }
   if (a==8) {
      if (m < -1.15) return 0;
      if (m < -0.67) return 1;
      if (m < -0.32) return 2;
      if (m < 0) return 3;
      if (m < 0.32) return 4;
      if (m < 0.67) return 5;
      if (m < 1.15) return 6;
      return 7;
   }
   if (a==9) {
      if (m < -1.22) return 0;
      if (m < -0.76) return 1;
      if (m < -0.43) return 2;
      if (m < -0.14) return 3;
      if (m < 0.14) return 4;
      if (m < 0.43) return 5;
      if (m < 0.76) return 6;
      if (m < 1.22) return 7;
      return 8;
   }
   if (a==10) {
      if (m < -1.28) return 0;
      if (m < -0.84) return 1;
      if (m < -0.52) return 2;
      if (m < -0.25) return 3;
      if (m < 0) return 4;
      if (m < 0.25) return 5;
      if (m < 0.52) return 6;
      if (m < 0.84) return 7;
      if (m < 1.28) return 8;
      return 9;
   }
}


void readStringVector(ifstream &in, string tag, vector <string> &list) {
   tag.insert(1, "/");
   list.clear();
   string s;
   while (s != tag && in) {
      in>>s;
      if (s != tag) list.push_back(s);
   }
}

void readIntVector(ifstream &in, string tag, vector <int> &list) {
   tag.insert(1, "/");
   list.clear();
   string s;
   while (s != tag && in) {
      in>>s;
      if (s != tag) list.push_back(atoi(s.c_str()));
   }
}

void readDoubleVector(ifstream &in, string tag, vector <double> &list) {
   tag.insert(1, "/");
   list.clear();
   string s;
   while (s != tag && in) {
      in>>s;
      if (s != tag) list.push_back(atof(s.c_str()));
   }
}

void breakSpaces(const string &s, vector <string> &ss) {
   ss.clear();
   string a;
   for (int i=0; i<s.size(); ++i) {
      if (s[i]==' ') {
	 if (a != "") {
	    ss.push_back(a);
	    a="";
	 }
      }
      else {
	 a.push_back(s[i]);
      }
   }
   if (a != "") ss.push_back(a);
}
