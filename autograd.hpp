#include "tensor.hpp"
#include <set>

class Value;
class Op {
   public:
      std::string name = "";
      virtual void forward(Value* out, std::vector<Value*>& inputs) = 0;
      virtual void backward(Value* out, std::vector<Value*>& inputs) = 0;
      
};

class Value{
      Op* op = nullptr;
   public:
      Tensor data;
      std::vector<Value*> children;
      Tensor grad = Tensor();
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
      void forward();
};     
    
Value* train(Value* model, std::vector<Tensor> inputs, std::vector<Tensor> outputs, int nSteps);



struct SigmoidOP: public Op {
   std::string name = "sigmoid";
   void forward(Value* out, std::vector<Value*>& inputs){
      out->data = inputs[0]->data.sigmoid();
   }
   void backward(Value* out, std::vector<Value*>& inputs){
      inputs[0]->grad = inputs[0]->data.sigmoidDeriv() * out->grad;
   }
};


struct MmOP: public Op {
   std::string name = "matmult";
   void forward(Value* out, std::vector<Value*>& inputs){
      out->data = inputs[0]->data.mm(inputs[1]->data);
   }
   void backward(Value* out, std::vector<Value*>& inputs){
      inputs[0]->grad = out->grad.mm(inputs[1]->data.transpose()) ;
      inputs[1]->grad = inputs[0]->data.transpose().mm(out->grad);
   }
};


struct AddOP: public Op {
   std::string name = "add";
   void forward(Value* out, std::vector<Value*>& inputs){
      out->data = inputs[0]->data + inputs[1]->data;
   }
   void backward(Value* out, std::vector<Value*>& inputs){
      inputs[0]->grad = out->grad;
      inputs[0]->grad = out->grad;
   }
};


struct MultOP: public Op {
   std::string name = "mult";
   void forward(Value* out, std::vector<Value*>& inputs){
      out->data = inputs[0]->data * inputs[1]->data;
   }
   void backward(Value* out, std::vector<Value*>& inputs){
      inputs[0]->grad = out->grad * inputs[1]->data;
      inputs[1]->grad = out->grad * inputs[0]->data;
   }
};
