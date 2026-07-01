#include "tensor.hpp"
#include <set>

class Value;
class Op {
   public:
      std::string name = "";
      virtual void forward(Value* out, std::vector<Value*>& inputs) = 0;
      virtual void backward(Value* out, std::vector<Value*>& inputs) = 0;
      
};
struct NoOP: public Op {
   std::string name = "no op";
   void forward(Value* out, std::vector<Value*>& inputs){
   }
   void backward(Value* out, std::vector<Value*>& inputs){

   }
};

class Value{
      Op* op = new NoOP();
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
      Value* sub(Value* b);
      Value* mult(Value* b);
      Value* mm(Value* b);
      Value* sigmoid();
      void topo_sort(std::set<Value*>* visited, std::vector<Value*>* sorted);
      void backward();
      void forward();
};

using LossFn = std::function<Value*(Value*, Value*)>;





struct SigmoidOP: public Op {
   std::string name = "sigmoid";
   void forward(Value* out, std::vector<Value*>& inputs){
      out->data = inputs[0]->data.sigmoid();
   }
   void backward(Value* out, std::vector<Value*>& inputs){
      inputs[0]->grad = inputs[0]->data.sigmoidDeriv() * out->grad + inputs[0]->grad;
   }
};


struct MmOP: public Op {
   std::string name = "matmult";
   void forward(Value* out, std::vector<Value*>& inputs){
      out->data = inputs[0]->data.mm(inputs[1]->data);
   }
   void backward(Value* out, std::vector<Value*>& inputs){
      inputs[0]->grad = out->grad.mm(inputs[1]->data.transpose()) + inputs[0]->grad;
      inputs[1]->grad = inputs[0]->data.transpose().mm(out->grad)+ inputs[1]->grad;
   }
};


struct AddOP: public Op {
   std::string name = "add";
   void forward(Value* out, std::vector<Value*>& inputs){
      out->data = inputs[0]->data + inputs[1]->data;
   }
   void backward(Value* out, std::vector<Value*>& inputs){
      inputs[0]->grad = out->grad + inputs[0]->grad;
      inputs[0]->grad = out->grad + inputs[1]->grad;
   }
};

struct SubOP: public Op {
   std::string name = "sub";
   void forward(Value* out, std::vector<Value*>& inputs){
      out->data = inputs[0]->data - inputs[1]->data;
   }
   void backward(Value* out, std::vector<Value*>& inputs){
      inputs[0]->grad = out->grad + inputs[0]->grad;
      inputs[0]->grad = -out->grad + inputs[1]->grad;
   }
};

struct MultOP: public Op {
   std::string name = "mult";
   void forward(Value* out, std::vector<Value*>& inputs){
      out->data = inputs[0]->data * inputs[1]->data;
   }
   void backward(Value* out, std::vector<Value*>& inputs){
      inputs[0]->grad = out->grad * inputs[1]->data + inputs[0]->grad;
      inputs[1]->grad = out->grad * inputs[0]->data + inputs[1]->grad;
   }
};


class Model{
public:
   virtual std::vector<Value*> params() = 0;
   virtual Value* create(Value* inputs) = 0;
   
    void zeroGrad(){
       for(Value* v: this->params()){
          v->grad = Tensor();
       }
    }
};

class MLP : public Model {
public:
    Value* w1;
    Value* w2;
    Value* w3;

    MLP() {
        w1 = new Value(Tensor({784,784}), "w1");
        w2 = new Value(Tensor({100,784}), "w2");
        w3 = new Value(Tensor({10,100}), "w3");
        w1->data.fillRandom();
        w2->data.fillRandom();
        w3->data.fillRandom();
    }

    Value* create(Value* input) override {
        return w3->mm(w2->mm(w1->mm(input)->sigmoid())->sigmoid())->sigmoid();
    }

    std::vector<Value*> params() override {
        return {w1, w2, w3};
    }

};


void train(Model& model, std::vector<Value*> inputs, std::vector<Value*> targets, int nSteps, LossFn loss_fun);
