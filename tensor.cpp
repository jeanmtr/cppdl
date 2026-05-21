#include "tensor.hpp"
#include <algorithm>

Tensor::Tensor(const std::vector<int>& shape){
   //jsp si faut pas bloquer le cas {}
   this->shape = shape;
   for(int i = 0;i<shape.size();i++){
      if(i==0){
         stride.push_back(1);
      }
      else{
         stride.push_back(stride[i-1]*shape[i-1]);
      }
   }
   size = shape.empty() ? 1 : stride.back()*shape.back();
   data = std::make_shared<std::vector<double>>(size, 0.0);
}

Tensor::Tensor(){
   this->shape = {};
   this->stride = {};
   this->size = 1;
   data = std::make_shared<std::vector<double>>(1, 0.0);
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
   return (*data)[sum];
}

const double& Tensor::get(const std::vector<int>& pos) const{
   int dim = pos.size();
   assert(dim == this->shape.size());
   for(int i = 0; i<dim;i++){
      assert(pos[i]<this->shape[i]);
   }
   int sum = 0;
   for(int i = 0;i<dim;i++){
      sum += pos[i]*this->stride[i];
   }
   return (*data)[sum];
}
//for scalars
double& Tensor::get(){
   assert(this->size == 1);
   return (*data)[0];

}

const double& Tensor::get() const {
   assert(this->size == 1);
   return (*data)[0];
}



template <typename Func> 
void iterate(const std::vector<int>& shape, const std::vector<int>& stride, Func&& fn){
   int ndim = shape.size();
   std::vector<int> x(ndim,0);
   int offset = 0;
   while(true){
      fn(offset,x);
      
      int dim = ndim - 1;
      while(dim>=0){
         x[dim]++;
         offset += stride[dim];
         if (x[dim] < shape[dim]){
            break;
         }
         offset -= shape[dim] * stride[dim];
         x[dim] = 0;
         dim --;
      }
      if(dim < 0){
         break;
      }
   }
}
//this is a specific case but idk if broader case is needed.
//i should use exceptions but idk how to use them yet
Tensor Tensor::broadcast(const std::vector<int>& shape) const{
   int dim = this->shape.size();
   int newDim = shape.size();
   bool oneFlag = false;
   for(int i = 0; i < newDim;i++){
      if (i < dim){
         if (this->shape[dim-1] == 1)
            oneFlag = true;
         else
            assert(this->shape[dim-i] == shape[newDim - i ] && !oneFlag);
      }
   }
   Tensor out(shape);
   out.data = this->data;
   return out;
}

//TODO: do try catch block for broadcasting, 
//rn we only broadcast other so first arg will always be the final shape
Tensor Tensor::operator+(const Tensor& other){
   int other_dim = other.shape.size();
   Tensor broadcasted = other;
   if(other_dim == this->shape.size()){
      for(int i = 0; i < other_dim;i++){
         if(other.shape[i] != this->shape[i]){
            broadcasted = other.broadcast(this->shape);
            break;
         }
      }
   }
   else
      broadcasted = other.broadcast(this->shape);
   Tensor out(this->shape);
   iterate(this->shape,this->stride, [&](int offset, const std::vector<int>& x){
      out.get(x) = this->get(x) + other.get(x);
   });
   return out; 
}

Tensor Tensor::operator*(const Tensor& other){
   int other_dim = other.shape.size();
   assert(other_dim == this->shape.size());
   for(int i = 0; i < other_dim;i++){
      assert(other.shape[i] == this->shape[i]);
   }
   Tensor out(this->shape);
   iterate(this->shape,this->stride, [&](int offset, const std::vector<int>& x){
      out.get(x) = this->get(x) * other.get(x);
   });
   return out; 
}

Tensor Tensor::mm(const Tensor& other){
   assert(this->shape.size() == 2 && other.shape.size() == 2);
   int n = this->shape[0];
   assert(this->shape[1] == other.shape[0]);
   Tensor out({this->shape[1],other.shape[1]});
   iterate(out.shape,out.stride,[&](int offset,const std::vector<int>& x){
      for(int i = 0;i<n;i++){
         out.get(x) += this->get({x[0],i})*other.get({i,x[1]});
      }
   });
   return out;
}

Tensor Tensor::sigmoid(){
   Tensor out(this->shape);
   iterate(out.shape,out.stride,[&](int offset, const std::vector<int>& x){
      out.get(x) = 1/(1 + exp(-this->get(x)));
   });
   return out;
}

Tensor Tensor::sigmoidDeriv(){
   Tensor out(this->shape);
   iterate(out.shape,out.stride,[&](int offset, const std::vector<int>& x){
      out.get(x) = exp(-this->get(x))/pow(1 + exp(-this->get(x)),2);
   });
   return out;
}
//not sure this makes a reference to data.
Tensor Tensor::transpose(){
   assert(this->shape.size() == 2 );
   Tensor out({this->shape[1],this->shape[0]});
   out.stride = {this->stride[1],this->stride[0]};
   out.data = this->data;
   return out;
}
void Tensor::printShape(){
      std::cout << "[";
      for(const int& i: this->shape){
         std::cout << i;
      }
      std::cout << "]";
}

int main(){
   Tensor aaaa({2,2});
   std::cout << "nice \n";
}

