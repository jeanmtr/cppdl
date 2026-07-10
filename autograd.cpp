#include "autograd.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>

void Value::print() {
  std::cout << "[" << label << "]: ";
  std::cout << " data =";
  if (data.shape.size() == 0) {
    std::cout << data.get({});
  } else
    this->data.printShape();
  std::cout << ", grad =";
  if (grad.shape.size() == 0) {
    std::cout << grad.get({});
  } else
    this->grad.printShape();
  std::cout << ", childrens = " << children.size();
  std::cout << ", op =" << op->name;
  std::cout << "\n";
}
// there must be a way to keep track of visited nodes better
void Value::topo_sort(std::set<Value *> *visited,
                      std::vector<Value *> *sorted) {
  visited->insert(this);
  std::vector<Value *>::iterator itr;
  for (itr = children.begin(); itr != children.end(); itr++) {
    if (!visited->count(*itr)) {
      (*itr)->topo_sort(visited, sorted);
    }
  }
  sorted->push_back(this);
}
void Value::backward() {
  auto start = std::chrono::steady_clock::now();
  std::set<Value *> visited;
  std::vector<Value *> sorted;
  this->topo_sort(&visited, &sorted);
  // for (Value* v : sorted) v->grad = Tensor(); // set to 0, this might cause
  // brodcasting issues later
  this->grad = Tensor();
  grad.get() = 1.0;
  std::reverse(sorted.begin(), sorted.end());
  auto end = std::chrono::steady_clock::now();
  auto bw_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "bw duration (ms): " << bw_duration.count() << "\n";
  for (Value *v : sorted) {
    // v->print();
    v->op->backward(v, v->children);
  }
}

void Value::forward() {
  std::set<Value *> visited;
  std::vector<Value *> sorted;
  this->topo_sort(&visited, &sorted);
  for (Value *v : sorted) {
    v->op->forward(v, v->children);
    // v->print();
  }
}

Value *Value::mult(Value *b) {
  Value *out = new Value(Tensor());
  out->op = new MultOP;
  out->children = {this, b};
  return out;
}

Value *Value::add(Value *b) {
  Value *out = new Value(Tensor());
  out->op = new AddOP;
  out->children = {this, b};
  return out;
}

Value *Value::sub(Value *b) {
  Value *out = new Value(Tensor());
  out->op = new SubOP;
  out->children = {this, b};
  return out;
}

Value *Value::mm(Value *b) {
  Value *out = new Value(Tensor());
  out->op = new MmOP;
  out->children = {this, b};
  return out;
}

Value *Value::sigmoid() {
  Value *out = new Value(Tensor());
  out->op = new SigmoidOP();
  out->children = {this};
  return out;
}
Value *Value::sum() {
  Value *out = new Value(Tensor());
  out->op = new SumOP();
  out->children = {this};
  return out;
}
Value *Value::power(double expo) {
  Value *out = new Value(Tensor());
  out->op = new PowerOP(expo);
  out->children = {this};
  return out;
}
// TODO: implement batching bc this is sad
void train(Model &model, Value* inputs,
           Value* targets, int nSteps, LossFn loss_fun) {
  auto start = std::chrono::steady_clock::now();
  double alpha = 0.0005;
  Value *output =
      model.create(inputs); // this does not work, faut changer
  Value *loss = loss_fun(output, targets);
  for (int i = 0; i < nSteps; i++) {
    std::cout << "############## step no :" << i << "\n";
    model.zeroGrad();

      auto start_time = std::chrono::steady_clock::now();

      auto creation_time = std::chrono::steady_clock::now();
      loss->forward();
      auto forward_time = std::chrono::steady_clock::now();

      std::cout << "loss for input " << "is :" << loss->data.get()
                << "guess was : [";
      /*
      int target_val = 0;
      for (int k = 0; k < 10; k++) {
        std::cout << output->data.get({k, 0}) << ",";
        if (targets[j]->data.get({k, 0}) == 1)
          target_val = k;
      }
      
      std::cout << "] expected was: " << target_val << "\n";
      */
      loss->backward();
      auto backward_time = std::chrono::steady_clock::now();

      auto creation_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(creation_time -
                                                                start_time);
      auto forward_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(forward_time -
                                                                creation_time);
      auto backward_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(backward_time -
                                                                forward_time);
      // std::cout << "creation took: " << creation_duration.count() << "forward
      // took : " << forward_duration.count() << "backward took : " <<
      // backward_duration.count() << "\n";
    
    for (Value *v : model.params()) {
      v->data = v->data - v->grad * alpha;
    }

    auto end = std::chrono::steady_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Training took " << duration.count() << " ms." << "\n";
  }
}

struct Dataset {
  Value* training_in;
  Value* training_out;
  Value* test_in;
  Value* test_out;
};

Dataset mnistToValue(int trainLen) {
  std::cout << "[+] Reading inputs \n";
  Dataset result;
  std::ifstream file("./mnist.csv");
  std::string line;
  int trainingLen = 100;
  std::getline(file, line); // skip header row if present
  int lineCount = 0;

  result.training_in = new Value(Tensor({trainLen, 784, 1}));
  result.training_out= new Value(Tensor({trainLen, 10, 1}));
  for (int i = 0; i < trainLen; i++) {
    if (!std::getline(file, line)) {
      std::cout << "error, end of file \n";
      exit(1);
    }
    std::stringstream ss(line);
    std::string cell;
    bool firstCol = true;
    lineCount++;
    int pxCount = 0;
    while (std::getline(ss, cell, ',')) {

      if (pxCount == 784) {
        int num = std::stoi(cell);
        if (lineCount < trainingLen) {
          result.training_out->data.get({i,num, 0}) = 1.0;
        }

      }

      else {
        result.training_in->data.get({i,pxCount, 0}) = std::stoi(cell) / 255.0;
      }
      pxCount++;
    }
  }

  std::cout << "[+] successfully read csv \n";
  return result;
}

void testTrain() {
  Dataset ds = mnistToValue(100);
  LossFn mse = [](Value *pred, Value *target) {
    Value *diff = pred->sub(target)->power(2)->sum();
    return diff; // (pred - target)²
  };
  std::cout << "[+] defining the model \n";
  MLP model;
  std::cout << "[+] sucessfully defined the model, beginning the training \n";

  train(model, ds.training_in, ds.training_out, 100, mse);
}

int main() {
  testTrain();
  std::cout << "[+] youpi \n";
  return 0;
}
