#include <bits/stdc++.h>
#include "autograd.hpp"


void Value::print(){
   if (op.empty()){
      cout << "["<< label << "]: ";
      cout << " data =" << data;
      cout << ", grad =" << grad;
      cout << ", childrens = " << children.size();
      cout << "\n";
   }
   else {
   cout << "["<< label << "]: ";
   cout << " data =" << data;
   cout << ", grad =" << grad;
   cout << ", childrens = " << children.size();
   cout << ", op =" << op;
   cout << "\n";
   }
}
//there must be a way to keep track of visited nodes better
void Value::topo_sort(set<Value*>* visited, vector<Value*>* sorted){
   visited->insert(this);
   set<Value*>::iterator itr;
   for(itr = children.begin();itr!=children.end();itr++){
      if(!visited->count(*itr)){
         (*itr)->topo_sort(visited, sorted);
      }
   }
   sorted->push_back(this);
}
void Value::backward(){
   grad = 1;
   set<Value*> visited;
   vector<Value*> sorted;
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
   cout << "we applied \n";
}

Value* Value::add(Value* b){
   Value* out = new Value(data + b->data);
   out->op = "+";
   out->_backward = new AddBackward(this,b,out);
   out->children = {this,b};
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
   cout << "[+] youpi \n";
   return 0;
}
