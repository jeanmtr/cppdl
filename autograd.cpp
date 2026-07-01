#include "autograd.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <chrono>

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
   //for (Value* v : sorted) v->grad = Tensor(); // set to 0, this might cause brodcasting issues later
   this->grad = Tensor();
   grad.get() = 1.0;
   std::reverse(sorted.begin(),sorted.end());
   for(Value* v: sorted){
      //v->print();
      v->op->backward(v,v->children);
   }
}

void Value::forward(){
   std::set<Value*> visited;
   std::vector<Value*> sorted;
   this->topo_sort(&visited,&sorted);
   for(Value* v: sorted){
      v->op->forward(v,v->children);
      //v->print();
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
Value* Value::sum(){
   Value* out = new Value(Tensor());
   out->op = new SumOP();
   out->children = {this};
   return out;
}
Value* Value::power(double expo){
   Value* out = new Value(Tensor());
   out->op = new PowerOP(expo);
   out->children = {this};
   return out;
}
// TODO: implement batching bc this is sad
void train(Model& model, std::vector<Value*> inputs, std::vector<Value*> targets, int nSteps, LossFn loss_fun){
   auto start = std::chrono::steady_clock::now();
   double alpha = 0.0005;
   Value* current_input = inputs[0];
   Value* output = model.create(current_input); //this does not work, faut changer
   Value* loss = loss_fun(output,targets[0]);
   for (int i = 0;i < nSteps; i++){
      std::cout << "############## step no :" << i << "\n";
      model.zeroGrad();
      for (int j = 0; j < inputs.size(); j++){

         auto start_time = std::chrono::steady_clock::now();
         output = model.create(inputs[j]);
         
         loss = loss_fun(output,targets[j]);
         auto creation_time = std::chrono::steady_clock::now();
         loss->forward();
         auto forward_time = std::chrono::steady_clock::now();
         std::cout << "loss for input " << j << "is :" << loss->data.get() << "guess was : [";
         int target_val = 0;
         for(int k = 0; k < 10; k++){
            std::cout << output->data.get({k,0}) << ",";
            if (targets[j]->data.get({k,0}) == 1)
               target_val = k;
         }
         std::cout << "] expected was: " << target_val << "\n";
         loss->backward();
         auto backward_time = std::chrono::steady_clock::now();


         auto creation_duration = std::chrono::duration_cast<std::chrono::milliseconds>(creation_time - start_time);
         auto forward_duration = std::chrono::duration_cast<std::chrono::milliseconds>(forward_time - creation_time);
         auto backward_duration = std::chrono::duration_cast<std::chrono::milliseconds>(backward_time - forward_time);
         std::cout << "creation took: " << creation_duration.count() << "forward took : " << forward_duration.count() << "backward took : " << backward_duration.count() << "\n";
   
      }
      for(Value* v: model.params()){
         v->data = v->data - v->grad * alpha;
      }

   auto end = std::chrono::steady_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
   std::cout << "Training took " << duration.count() << " ms." << "\n";
   }
}

struct Dataset{
   std::vector<Value*> training_in;
   std::vector<Value*> training_out;
   std::vector<Value*> test_in;
   std::vector<Value*> test_out;   
};


Dataset mnistToValue(){
   std::cout << "[+] Reading inputs \n";
   Dataset result;
   std::ifstream file("./mnist.csv");
   std::string line;
   int trainingLen = 100;
   std::getline(file, line); // skip header row if present
   int lineCount = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        bool firstCol = true;
        lineCount ++;
        Value* img = new Value(Tensor({784, 1}));
        Value* numVector = new Value(Tensor({10, 1}));
        int pxCount = 0;
        while (std::getline(ss, cell, ',')) {
           
            if (pxCount == 784) {
               int num = std::stoi(cell);
               if(lineCount < trainingLen){
                  numVector->data.get({num, 0}) = 1.0;
                  result.training_out.push_back(numVector);
                  }
               else{
                  numVector->data.get({num, 0}) = 1.0;
                  result.test_out.push_back(numVector);
                  }
                  
            }

            else {
                img->data.get({pxCount, 0}) = std::stoi(cell) / 255.0;
            }
            pxCount ++;
        }
         
         if(lineCount < trainingLen)
            result.training_in.push_back(img);
         else
            result.test_in.push_back(img);
    }
    std::cout << "[+] successfully read csv \n";
    return result;
}


void testTrain(){
   Dataset ds = mnistToValue();
   LossFn mse = [](Value* pred, Value* target) {
      Value* diff = pred->sub(target)->power(2)->sum();
      return diff;  // (pred - target)²
};
   std::cout << "[+] defining the model \n"; 
   MLP model;
   std::cout << "[+] sucessfully defined the model, beginning the training \n"; 
   
   train(model, ds.training_in, ds.training_out, 10, mse);
   
}



int main(){
   testTrain();
   std::cout << "[+] youpi \n";
   return 0;
}


