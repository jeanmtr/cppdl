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
   
   vector<Value*>::reverse_iterator itr;
   for(itr = sorted.rbegin();itr != sorted.rend();itr++){
      (*itr)->_backward->apply();
   }
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



int main(){
   Value a(10., "a");
   a.print();
   Value b(33.,"b");
   b.print();
   Value* c = a.add(&b);
   c->backward();
   a.print();
   cout << "[+] youpi \n";
   return 0;
}
