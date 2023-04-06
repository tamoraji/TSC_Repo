#ifndef _UTIL_H
#define _UTIL_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cmath>

using namespace std;

double mean(const vector <double> &v);

double stdDev(const vector <double> &v);

void normalize(vector <double> &v); // normalizes the vector to have zero mean and 1 standard deviation

int baseNum(const vector <int> &d, int a); // returns the number in the base a from digits in d

double alphabet(double m, int a); // gives the appropriate alphabet using the Guassian lookup table

double dotProduct(const vector <double> &v1, const vector <double> &v2);


template <typename T> 
void writeVector(ostream &out, const vector <T> &v, const string &c=" ") {
   for (int i=0; i<v.size(); ++i) out<<v[i]<<c;
   out<<"\n";
}

// used to convert an int or a double to string
template <typename T>
string itos(T n) {  
   std::ostringstream ost;
   ost<<n;
   return ost.str();
}

void readStringVector(ifstream &in, string tag, vector <string> &list);
void readIntVector(ifstream &in, string tag, vector <int> &list);
void readDoubleVector(ifstream &in, string tag, vector <double> &list);

template <typename T>
void readElement(ifstream &in, string tag, T &read) {
   in>>read;
   string s;
   in>>s;
   tag.insert(1, "/");
   assert(s==tag);
}

void breakSpaces(const string &s, vector <string> &ss);

#endif
