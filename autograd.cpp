#include "autograd.hpp"
#include <algorithm>


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
      if (!op.empty()){
   std::cout << ", op =" << op;
   }
   std::cout << "\n";
   
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
      v->print();
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
   std::cout << "begin mm bw \n";

   std::cout << "shape 1: ";
   for (int i: b->data.shape){std::cout << i << ",";}
   std::cout << "\n" << "shape 2: ";
   for (int i: out->grad.shape){std::cout << i << ",";}
   std::cout << "\n";
   a->grad = out->grad.mm(b->data.transpose()) ;
   b->grad = a->data.transpose().mm(out->grad);
}

Value* Value::mm(Value* b){
   Value* out = new Value(data.mm(b->data));
   out->op = "mm";
   out->_backward = new MMBackward(this, b, out);
   out->children = {this,b};
   return out;
}

void SigmoidBackward::apply(){
   std::cout << "begin sigmoid backwards \n";
   a->grad = a->data.sigmoidDeriv() * out->grad;

}

Value* Value::sigmoid(){
   Value* out = new Value(data.sigmoid());
   out->op = "sigmoid";
   out->_backward = new SigmoidBackward(this, out);
   out->children = {this};
   return out;
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
   out->print();
   std::cout << "we be done with the graph \n";
   w1.data.fillRandom();
   w2.data.fillRandom();
   w3.data.fillRandom();
   w4.data.fillRandom();

   std::cout << "arrays filled with rdm values \n";
   out->backward();
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
   testMlp();
   return 0;
}


