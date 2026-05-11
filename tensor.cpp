#include "tensor.hpp"

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
   data = std::vector<double>(size, 0.0);
}

Tensor::Tensor(){
   this->shape = {};
   this->stride = {};
   this->size = 1;
   data = std::vector<double>(1,0.0);
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
   return this->data[sum];
}
//for scalars
double& Tensor::get(){
   assert(this->size == 1);
   return this->data[0];
}

const double& Tensor::get() const {
   assert(this->size == 1);
   return this->data[0];
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


//pr le moment on aditionne que les tenseurs de mm shape
//faudrait implémenter le brodcasting pour des tenseurs pas de mm shape
Tensor Tensor::operator+(const Tensor& other){
   int other_dim = other.shape.size();
   assert(other_dim == this->shape.size());
   for(int i = 0; i < other_dim;i++){
      assert(other.shape[i] == this->shape[i]);
   }
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

