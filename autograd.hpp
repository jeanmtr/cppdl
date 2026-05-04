#include <bits/stdc++.h>
using namespace std;

class BackwardFn {
   public:
      virtual void apply(){};
};

class Value{
      BackwardFn* _backward = new BackwardFn();
   public:
      double data;
      set<Value*> children;
      double grad = 0;
      string op = "";
      string label;
      

      Value(double data, string label = "") : data(data), label(label){};

      void print();
      Value* add(Value* b);
      Value* mult(Value* b);
      void topo_sort(set<Value*>* visited, vector<Value*>* sorted);
      void backward();
};     
    



class AddBackward : public BackwardFn {
   Value* a;
   Value* b;
   Value* out;
   public:
      AddBackward(Value* a, Value* b, Value* out) : a(a), b(b), out(out) {}
      void apply() override;
};


class MultBackward: public BackwardFn {
   Value* a;
   Value* b;
   Value* out;
   public:
      MultBackward(Value* a, Value* b, Value* out) : a(a), b(b), out(out) {}
      void apply() override;
};

