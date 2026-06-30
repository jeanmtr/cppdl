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
   dim = shape.empty() ? 1 : stride.back()*shape.back();
   data = std::make_shared<std::vector<double>>(dim, 0.0);
}

Tensor::Tensor(){
   this->shape = {};
   this->stride = {};
   this->dim = 1;
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
   assert(this->dim == 1);
   return (*data)[0];

}

const double& Tensor::get() const {
   assert(this->dim == 1);
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
// there might be an issue if dim > newDim but idk how that might happen
Tensor Tensor::broadcast(const std::vector<int>& shape) const{
   std::cout << "attempting to broadcast tensors of shape: \n";
   std::cout << "shape 1 (the one being broadcasted): ";
   for (int i: this->shape){std::cout << i << ",";}
   std::cout << "\n" << "shape 2: ";
   for (int i: shape){std::cout << i << ",";}
   std::cout << "\n";
   int dim = this->shape.size();
   int newDim = shape.size();
   std::cout << dim << "," << newDim << "\n";
   bool oneFlag = false;
   int firstOne = newDim-dim;
   for(int i = 1; i <= dim;i++){
      if (this->shape[dim-i] == shape[newDim - i ] && !oneFlag){std::cout<<"a\n";}
      else if(this->shape[dim-i] == 1){ //why am i checking this
         if (!oneFlag){
            std::cout << "b\n";
            oneFlag = true;
            firstOne = dim -i + 1;
         }
      }
      else {
         std::cout << "could not broadcast tensors. \n ";
         std::exit(1);
      }
   }
   Tensor out(shape); //there might be a problem with the copy TODO
   out.data = this->data;
   for(int i = 0; i < firstOne; i++){
      out.stride[i] = 0;
   }
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
      out.get(x) = this->get(x) + broadcasted.get(x);
   });
   return out; 
}

Tensor Tensor::operator*(const Tensor& other){
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
      out.get(x) = this->get(x) * broadcasted.get(x);
   });
   return out; 
}

//TODO implement matrix multiplication but for tensors (repeating matrix mult)
Tensor Tensor::mm(const Tensor& other){
   std::cout << "doing mm \n";
   assert(this->shape.size() == 2 && other.shape.size() == 2);
   int n = this->shape[1];
   assert(this->shape[1] == other.shape[0]);
   Tensor out({this->shape[0],other.shape[1]});
   iterate(out.shape,out.stride,[&](int offset,const std::vector<int>& x){
      for(int i = 0;i<n;i++){
         //std::cout << x[0] << "," << x[1] << "," << i << "\n";
         out.get(x) += this->get({x[0],i})*other.get({i,x[1]});
      }
   });
   return out;
}

Tensor Tensor::power(double expo){
   Tensor out(this->shape);
   iterate(out.shape,out.stride,[&](int offset, const std::vector<int>& x){
      out.get(x) = pow(this->get(x),expo);
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
   std::cout << "end of deriv\n";
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


void Tensor::fillRandom(){
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    iterate(this->shape,this->stride, [&](int offset, const std::vector<int>& x){
       std::cout << x[0] << "," << x[1] << "\n";
       this->get(x) = dist(rng);
   });
   std::cout << "\n";
}

