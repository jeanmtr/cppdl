#include <bits/stdc++.h>
#include "autograd.hpp"

using namespace std;

class Value{
      BackwardFn* _backward;
   public:
      double data;
      set<Value*> children;
      double grad;
      string op;
      string label;
      
      Value(){
         data = 0.;
         grad = 0.;
         op = "";
         label = "";
   }

      Value(double da, string la = ""){
         data = da;
         label = la;
      }

      void print(){
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
      Value operator+(Value& b);
      Value operator*(Value& b);
      //there must be a way to keep track of visited nodes better
      void topo_sort(set<Value*>* visited, vector<Value*>* sorted){
         visited->insert(this);
         set<Value*>::iterator itr;
         for(itr = children.begin();itr!=children.end();itr++){
            if(!visited->count(*itr)){
               (*itr)->topo_sort(visited, sorted);
            }
         }
         sorted->push_back(this);
      }
      void backward(){
         grad = 1;
         set<Value*> visited;
         vector<Value*> sorted;
         this->topo_sort(&visited,&sorted);
         
         vector<Value*>::reverse_iterator itr;
         for(itr = sorted.rbegin();itr != sorted.rend();itr++){
            (*itr)->_backward->apply();
         }
      }
};     
    


class BackwardFn {
   virtual void apply() = 0;
};

class AddBackward : public BackwardFn {
   Value* a;
   Value* b;
   Value* out;

   AddBackward(Value* a, Value* b, Value* out) : a(a), b(b), out(out) {}
   
   void apply() override {
      a->grad = out->grad;
      b->grad = out->grad;
   }
};

Value Value::operator+(Value& b){
   Value out = Value(data + b.data);
   out.op = "+";
   out._backward = new AddBackward(this,&b,);
   out.children = {this,&b};
   return out;
}
class MultBackward : public BackwardFn {
   Value* a;
   Value* b;
   Value* out;
   
   void apply() override {
      a->grad = out->grad * b->data;
      b->grad = out->grad * a->data;
   }
};

Value Value::operator*(Value& b){
   Value out = Value(data * b.data);
   out.op = "*";
   out._backward = new MultBackward;
   out.children = {this,&b};
   return out;
}



int main(){
   Value a(10., "a");
   a.print();
   Value b(33.,"b");
   b.print();

   cout << "[+] youpi \n";
   return 0;
}
