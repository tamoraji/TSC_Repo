#ifndef _EXAMPLE_H
#define _EXAMPLE_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>

using namespace std;

class example { // time series example (independent of classifier)
   vector <double> v; // time series
   string label;
   int id; // index of the example in the data file
   string name; // dataset name
   bool ifTrain; // true if train example otherwise false

   public:
   example() {}

   example(const vector <double> &pv, const string &plabel, int pid, const string &pname, bool pifTrain) {
      v = pv;
      label = plabel;
      id = pid;
      name = pname;
      ifTrain = pifTrain;
   }

   string giveLabel() const {
      return label;
   }

   int giveId() const {
      return id;
   }
   string giveName() const {
      return name;
   }
   bool giveIfTrain() const {
      return ifTrain;
   }

   vector <double> giveV() const {
      return v;
   }
};

double compute_error(const vector <example> &e, const vector <string> &out_label);
double compute_error(const vector <string> &corr_label, const vector <string> &out_label);

#endif
