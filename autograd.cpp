#include "autograd.hpp"
#include <algorithm>


void Value::print(){
   if (op.empty()){
      std::cout << "["<< label << "]: ";
      std::cout << " data =";
      this->data.printShape();
      std::cout << ", grad =";
      this->grad.printShape();
      std::cout << ", childrens = " << children.size();
      std::cout << "\n";
   }
   else {
   std::cout << "["<< label << "]: ";
   std::cout << " data =";
   this->data.printShape();
   std::cout << ", grad =";
   this->grad.printShape();
   std::cout << ", childrens = " << children.size();
   std::cout << ", op =" << op;
   std::cout << "\n";
   }
}
//there must be a way to keep track of visited nodes better
void Value::topo_sort(std::set<Value*>* visited, std::vector<Value*>* sorted){
   visited->insert(this);
   std::set<Value*>::iterator itr;
   for(itr = children.begin();itr!=children.end();itr++){
      if(!visited->count(*itr)){
         (*itr)->topo_sort(visited, sorted);
      }
   }
   sorted->push_back(this);
}
void Value::backward(){
   grad = Tensor();
   grad.get() = 1.0;
   std::set<Value*> visited;
   std::vector<Value*> sorted;
   this->topo_sort(&visited,&sorted);
   std::reverse(sorted.begin(),sorted.end());
   for(Value* v: sorted){
      v->_backward->apply();
   }
}





void MultBackward::apply(){
   a->grad = out->grad * b->data;
   b->grad = out->grad * a->data;
}

Value* Value::mult(Value* b){
   Value* out = new Value(data * b->data);
   out->op = "*";
   out->_backward = new MultBackward(this,b,out);
   out->children = {this,b};
   return out;
}
void AddBackward::apply(){
   a->grad = out->grad;
   b->grad = out->grad;
}

Value* Value::add(Value* b){
   Value* out = new Value(data + b->data);
   out->op = "+";
   out->_backward = new AddBackward(this,b,out);
   out->children = {this,b};
   return out;
}

void MMBackward::apply(){
   a->grad = b->data.transpose() * out->grad;
   b->grad = a->data.transpose() * out->grad;
}

Value* Value::mm(Value* b){
   Value* out = new Value(data.mm(b->data));
   out->op = "mm";
   out->_backward = new MMBackward(this, b, out);
   out->children = {this,b};
   return out;
}

void SigmoidBackward::apply(){
   a->grad = a->data.sigmoidDeriv() * out->grad;

}

Value* Value::sigmoid(){
   Value* out = new Value(data.sigmoid());
   out->op = "sigmoid";
   out->_backward = new SigmoidBackward(this, out);
   out->children = {this};
   return out;
}


int main(){
   Value a(10., "a");
   a.print();
   Value b(33.,"b");
   b.print();
   Value* c = a.add(&b);
   Value d(100.,"d");

   Value f(1.33,"f");
   Value* e = d.add(&f);
   Value* g = c->mult(e);
   c->print();
   g->backward();


   a.print();
   b.print();
   c->print();
   d.print();
   e->print();
   g->print();
   f.print();
   std::cout << "[+] youpi \n";
   return 0;
}


