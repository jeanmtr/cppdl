#include "tensor.hpp"
#include <set>

class Value;
class Op {
   public:
      std::string name = "";
      double param;
      virtual void forward(Value* out, std::vector<Value*>& inputs) = 0;
      virtual void backward(Value* out, std::vector<Value*>& inputs) = 0;
      
};
struct NoOP: public Op {
   NoOP() {name = "leaf node";}
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
      Value* sum();
      Value* power(double expo);
      Value* sigmoid();
      void topo_sort(std::set<Value*>* visited, std::vector<Value*>* sorted);
      void backward();
      void forward();
      void reduce();
};






struct SigmoidOP: public Op {
   SigmoidOP(){name = "sigmoid";}
   void forward(Value* out, std::vector<Value*>& inputs){
      out->data = inputs[0]->data.sigmoid();
   }
   void backward(Value* out, std::vector<Value*>& inputs){
      inputs[0]->grad = inputs[0]->data.sigmoidDeriv() * out->grad + inputs[0]->grad;
   }
};
struct SumOP: public Op {
   SumOP(){name = "sum";}
   void forward(Value* out, std::vector<Value*>& inputs){
      out->data = inputs[0]->data.sum() ;
   }
   void backward(Value* out, std::vector<Value*>& inputs){
      inputs[0]->grad = Tensor({inputs[0]->data.shape}) + out->grad + inputs[0]->grad;
   }
};
struct PowerOP: public Op {
   PowerOP(double expo){name = "power"; param = expo;}
   void forward(Value* out, std::vector<Value*>& inputs){
      out->data = inputs[0]->data.power(param);
   }
   void backward(Value* out, std::vector<Value*>& inputs){
      inputs[0]->grad = inputs[0]->data.power(param - 1) * out->grad * param + inputs[0]->grad;
   }
};


struct MmOP: public Op {
   MmOP(){name = "matmult";}
   void forward(Value* out, std::vector<Value*>& inputs){
      out->data = inputs[0]->data.mm(inputs[1]->data);
   }
   void backward(Value* out, std::vector<Value*>& inputs){
      inputs[0]->grad = out->grad.mm(inputs[1]->data.transpose()) + inputs[0]->grad;
      inputs[1]->grad = inputs[0]->data.transpose().mm(out->grad)+ inputs[1]->grad;
   }
};


struct AddOP: public Op {
   AddOP(){name = "add";}
   void forward(Value* out, std::vector<Value*>& inputs){
      out->data = inputs[0]->data + inputs[1]->data;
   }
   void backward(Value* out, std::vector<Value*>& inputs){
      inputs[0]->grad = out->grad + inputs[0]->grad;
      inputs[1]->grad = out->grad + inputs[1]->grad;
   }
};

struct SubOP: public Op {
   SubOP(){name = "sub";}
   void forward(Value* out, std::vector<Value*>& inputs){
      out->data = inputs[0]->data - inputs[1]->data;
   }
   void backward(Value* out, std::vector<Value*>& inputs){
      inputs[0]->grad = out->grad + inputs[0]->grad;
      inputs[1]->grad = -out->grad + inputs[1]->grad;
   }
};

struct MultOP: public Op {
   MultOP() {name = "mult";}
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
    Value* b1;
    Value* b2;
    Value* b3;

    MLP() {
        w1 = new Value(Tensor({32,784}), "w1");
        w2 = new Value(Tensor({32,32}), "w1");
        w3 = new Value(Tensor({10,32}), "w3");
        b1 = new Value(Tensor({32,1}), "b1");
        b2 = new Value(Tensor({32,1}), "b2");
        b3 = new Value(Tensor({10,1}), "b3");
        w1->data.fillRandom();
        w2->data.fillRandom();
        w3->data.fillRandom();
        b1->data.fillRandom();
        b2->data.fillRandom();
        b3->data.fillRandom();
    }

    Value* create(Value* input) override {
        return w3->mm(w2->mm(w1->mm(input)->add(b1)->sigmoid())->add(b2)->sigmoid())->add(b3)->sigmoid();
    }

    std::vector<Value*> params() override {
        return {w1, w2, w3, b1, b2, b3};
    }

};


using LossFn = std::function<Value*(Value*, Value*, Model&)>;
void train(Model& model, std::vector<Value*> inputs, std::vector<Value*> targets, int nSteps, LossFn loss_fun);
