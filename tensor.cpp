#include "tensor.hpp"

Tensor::Tensor(const std::vector<int>& shape){
   this->shape = shape;
   for(int i = 0;i<shape.size();i++){
      if(i==0){
         stride.push_back(1);
      }
      else{
         stride.push_back(stride[i-1]*shape[i-1]);
      }
   }
   size = stride.back()*shape.back();
   data = std::vector<double>(size, 0.0);
}
  
double& Tensor::get(const std::vector<int>& pos){
   int dim = pos.size();
   assert(dim == this->shape.size());
   for(int i = 0; i<dim;i++){
      assert(pos[i]<this->shape[i]);
   }
   int sum = 0;
   for(int i = 0;i<dim;i++){
      sum += pos[i]*this->stride[i];
   }
   return this->data[sum];
   
}
//pr le moment on aditionne que les tenseurs de mm shape
Tensor Tensor::operator+(const Tensor& other){
   int other_dim = other.shape.size();
   assert(other_dim == this->shape.size());
   for(int i = 0; i < other_dim;i++){
      assert(other.shape[i] == this->shape[i]);
   }
   Tensor out(this->shape);
   //hoping the the compiler is optimizing this
   for(int i = 0;i < this->size;i++){
      out.data[i] = other.data[i] + this->data[i];
   }
   return out;
}

Tensor Tensor::operator*(const Tensor& other){
   int other_dim = other.shape.size();
   assert(other_dim == this->shape.size());
   for(int i = 0; i < other_dim;i++){
      assert(other.shape[i] == this->shape[i]);
   }
   Tensor out(this->shape);
   //hoping the the compiler is optimizing this
   for(int i = 0;i < this->size;i++){
      out.data[i] = other.data[i] * this->data[i];
   }
   return out;
}

int main(){
   Tensor aaaa({2,2});
   std::cout << "nice \n";
}

