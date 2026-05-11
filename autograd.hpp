#include "tensor.hpp"
#include <set>

class BackwardFn {
   public:
      virtual void apply(){};
};

class Value{
      BackwardFn* _backward = new BackwardFn();
   public:
      Tensor data;
      std::set<Value*> children;
      Tensor grad = Tensor();
      std::string op = "";
      std::string label;
      

      Value(Tensor data, std::string label = "") : data(data), label(label){};

      Value(double data, std::string label = "") : data(Tensor()), label(label){
         this->data.get() = data;
   };
      void print();
      Value* add(Value* b);
      Value* mult(Value* b);
      Value* mm(Value* b);
      Value* sigmoid();
      void topo_sort(std::set<Value*>* visited, std::vector<Value*>* sorted);
      void backward();
};     
    




class SigmoidBackward: public BackwardFn {
   Value* a;
   Value* out;
   public:
      SigmoidBackward(Value* a, Value* out) : a(a), out(out) {}
      void apply() override;
};
class MMBackward : public BackwardFn {
   Value* a;
   Value* b;
   Value* out;
   public:
      MMBackward(Value* a, Value* b, Value* out) : a(a), b(b), out(out) {}
      void apply() override;
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

