#include "autograd.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>

void Value::print(){
   std::cout << "["<< label << "]: ";
   std::cout << " data =";
   if (data.shape.size() == 0){
      std::cout << data.get({});
   }
   else
      this->data.printShape();
   std::cout << ", grad =";
   if (grad.shape.size() == 0){
      std::cout << grad.get({});
   }
   else
      this->grad.printShape();
   std::cout << ", childrens = " << children.size();
   std::cout << ", op =" << op->name;
   std::cout << "\n";
   
}
//there must be a way to keep track of visited nodes better
void Value::topo_sort(std::set<Value*>* visited, std::vector<Value*>* sorted){
   visited->insert(this);
   std::vector<Value*>::iterator itr;
   for(itr = children.begin();itr!=children.end();itr++){
      if(!visited->count(*itr)){
         (*itr)->topo_sort(visited, sorted);
      }
   }
   sorted->push_back(this);
}
void Value::backward(){
   std::set<Value*> visited;
   std::vector<Value*> sorted;
   this->topo_sort(&visited,&sorted);
   for (Value* v : sorted) v->grad = Tensor(); // set to 0, this might cause brodcasting issues later
   grad.get() = 1.0;
   std::reverse(sorted.begin(),sorted.end());
   for(Value* v: sorted){
      v->print();
      v->op->backward(v,v->children);
   }
}

void Value::forward(){
   std::set<Value*> visited;
   std::vector<Value*> sorted;
   this->topo_sort(&visited,&sorted);
   for(Value* v: sorted){
      v->print();
      v->op->forward(v,v->children);
   }
}





Value* Value::mult(Value* b){
   Value* out = new Value(Tensor());
   out->op = new MultOP;
   out->children = {this,b};
   return out;
}

Value* Value::add(Value* b){
   Value* out = new Value(Tensor());
   out->op = new AddOP;
   out->children = {this,b};
   return out;
}

Value* Value::sub(Value* b){
   Value* out = new Value(Tensor());
   out->op = new SubOP;
   out->children = {this,b};
   return out;
}

Value* Value::mm(Value* b){
   Value* out = new Value(Tensor());
   out->op = new MmOP;
   out->children = {this,b};
   return out;
}


Value* Value::sigmoid(){
   Value* out = new Value(Tensor());
   out->op = new SigmoidOP();
   out->children = {this};
   return out;
}
// TODO: implement batching bc this is sad
void train(Model& model, std::vector<Value*> inputs, std::vector<Value*> targets, int nSteps, LossFn loss_fun){
   double alpha = 0.1;
   Value* current_input = inputs[0];
   Value* output = model.create(current_input); //this does not work, faut changer
   Value* loss = loss_fun(output,targets[0]);
   for (int i = 0;i < nSteps; i++){
      std::cout << "############## step no :" << i << "\n";
      model.zeroGrad();
      for (int j = 0; j < inputs.size(); j++){
         output = model.create(inputs[j]);
         loss = loss_fun(output,targets[j]);
         loss->forward();
         std::cout << "loss for input " << j << "is :" << loss->data.get({0,0}) << "\n";
         loss->backward();
      }
      for(Value* v: model.params()){
         v->data = v->data + v->grad * alpha;
      }
      
   }
}

void testMlp(){
   Value input(Tensor({10,1}),"inputs");
   Value w1(Tensor({10,10}), "w1");
   Value w2(Tensor({7,10}), "w2");
   Value w3(Tensor({5,7}), "w3");
   Value w4(Tensor({1,5}), "w4");

   Value* l1 = w1.mm(&input)->sigmoid();
   Value* l2 = w2.mm(l1)->sigmoid();
   Value* l3 = w3.mm(l2)->sigmoid();
   Value* l4 = w4.mm(l3)->sigmoid();
   Value* out = l4->sigmoid();
   out->forward();
   out->print();
   std::cout << "we be done with the graph \n";
   w1.data.fillRandom();
   w2.data.fillRandom();
   w3.data.fillRandom();
   w4.data.fillRandom();

   std::cout << "arrays filled with rdm values \n";
   out->backward();
   out->print();
   std::cout << out->data.get({0,0}) << "\n";
}
struct Dataset{
   std::vector<Value*> training_in;
   std::vector<Value*> training_out;
   std::vector<Value*> test_in;
   std::vector<Value*> test_out;   
};


Dataset mnistToValue(){
   Dataset result;
   std::ifstream file("./mnist.csv");
   std::string line;
   int trainingLen = 1000;
   std::getline(file, line); // skip header row if present
   int lineCount = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        bool firstCol = true;
        lineCount ++;
        Value* img = new Value(Tensor({784}));
        Value* numVector = new Value(Tensor({10}));
        int pxCount = 0;
        while (std::getline(ss, cell, ',')) {
           
            if (firstCol) {
               if(lineCount < trainingLen){
                  numVector->data.get({std::stoi(cell)}) = 1.0;
                  result.training_out.push_back(numVector);
                  }
               else{
                  numVector->data.get({std::stoi(cell)}) = 1.0;
                  result.test_out.push_back(numVector);
                  }
                  
               firstCol = false;
            } else {
                img->data.get({pxCount}) = std::stof(cell) / 255.0;
                pxCount ++;
            }
        }
         
         if(lineCount < trainingLen)
            result.training_in.push_back(img);
         else
            result.test_in.push_back(img);
    }
    return result;
}


void testTrain(){
   Dataset ds = mnistToValue();
   LossFn mse = [](Value* pred, Value* target) {
      //here we assume that output shape is always a n*1*...*1 vector, this should be addressed
      int len = target->data.shape[0];
      Value addVal(Tensor({1,len}));
      addVal.data = addVal.data + 1;
      Value* diff = addVal.mm(pred->sub(target));
      return diff->mult(diff);  // (pred - target)²
};
   MLP model;
   train(model, ds.training_in, ds.training_out, 100, mse);
   
}



int main(){
   testTrain();
   std::cout << "[+] youpi \n";
   return 0;
}


