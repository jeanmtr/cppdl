#include "tensor.hpp"
#include <algorithm>
Tensor::Tensor(const std::vector<int> &shape) {
  // jsp si faut pas bloquer le cas {}
  this->shape = shape;
  for (int i = 0; i < shape.size(); i++) {
    assert(shape[i] != 0);
    if (i == 0) {
      stride.push_back(1);
    } else {
      stride.push_back(stride[i - 1] * shape[i - 1]);

    }
  }
  dim = shape.empty() ? 1 : stride.back() * shape.back();
  data = std::make_shared<std::vector<double>>(dim, 0.0);
}

Tensor::Tensor() {
  this->shape = {};
  this->stride = {};
  this->dim = 1;
  data = std::make_shared<std::vector<double>>(1, 0.0);
}

double &Tensor::get(const std::vector<int> &pos) {
  int dim = pos.size();
  assert(dim == this->shape.size());
  for (int i = 0; i < dim; i++) {
    assert(pos[i] < this->shape[i]);
  }
  int sum = 0;
  for (int i = 0; i < dim; i++) {
    sum += pos[i] * this->stride[i];
  }
  return (*data)[sum];
}

const double &Tensor::get(const std::vector<int> &pos) const {
  int dim = pos.size();
  assert(dim == this->shape.size());
  for (int i = 0; i < dim; i++) {
    assert(pos[i] < this->shape[i]);
  }
  int sum = 0;
  for (int i = 0; i < dim; i++) {
    sum += pos[i] * this->stride[i];
  }
  return (*data)[sum];
}
// for scalars
double &Tensor::get() {
  assert(this->dim == 1);
  return (*data)[0];
}

const double &Tensor::get() const {
  assert(this->dim == 1);
  return (*data)[0];
}

template <typename Func>
void iterate(const std::vector<int> &shape, const std::vector<int> &stride,
             Func &&fn) {
  int ndim = shape.size();
  std::vector<int> x(ndim, 0);
  int offset = 0;
  while (true) {
    fn(offset, x);

    int dim = ndim - 1;
    while (dim >= 0) {
      x[dim]++;
      offset += stride[dim];
      if (x[dim] < shape[dim]) {
        break;
      }
      offset -= shape[dim] * stride[dim];
      x[dim] = 0;
      dim--;
    }
    if (dim < 0) {
      break;
    }
  }
}

int Tensor::effectiveDim() const{
  int dim = this->shape.size();
  int trailingOnes = 0;
  while(this->shape[trailingOnes] == 1){
    trailingOnes ++;
  }
  return dim - trailingOnes;
}
// until is used to ignore the n last dims (ex: for matmul )
// this is not deffensive at all, shapes and strides should be cleared
// also until should be checked
void Tensor::broadcast(const Tensor& other, int until, Tensor* out1, Tensor* out2) const{
  //std::cout << "br en cours\n";
  int dimThis = this->shape.size();
  int dimOther= other.shape.size();
  int outDim = std::max(dimThis,dimOther);
  int smallDim = std::min(dimThis,dimOther);
  bool thisIsSmaller = dimOther==outDim;
  int thisOffset = outDim - dimThis;
  int otherOffset = outDim - dimOther;

  assert(smallDim >= until);


  if (thisIsSmaller){
    out1->stride.insert(out1->stride.begin(),thisOffset,0);
  }
  else{
    out2->stride.insert(out2->stride.begin(),otherOffset,0);
  }
  out1->stride.insert(out1->stride.end(),this->stride.begin(),this->stride.end());
  out2->stride.insert(out2->stride.end(),other.stride.begin(),other.stride.end());
  // first part
  for(int i = 0; i<outDim-smallDim; i++){
    if(thisIsSmaller){
      out1->shape.push_back(other.shape[i]);
      out2->shape.push_back(other.shape[i]);
    }
    else{
      out1->shape.push_back(this->shape[i]);
      out2->shape.push_back(this->shape[i]);
    }
  }
  //seconde part (main part)
  for(int i = outDim-smallDim; i<outDim-until; i++){
    int dim1,dim2;
      dim1 = this->shape[i-thisOffset];
      dim2 = other.shape[i-otherOffset];
      
    if (dim1 == 1){
      out1->shape.push_back(dim2);
      out2->shape.push_back(dim2);
      out1->stride[i] = 0;
    }
    else if (dim2== 1){
      out1->shape.push_back(dim1);
      out2->shape.push_back(dim1);
      out2->stride[i] = 0;
    }
    else if (dim1 == dim2){
      out1->shape.push_back(dim1);
      out2->shape.push_back(dim2);
    }
    else{
      std::cout << "could not broadcast shapes !" << dim1 << "!=" << dim2 << "\n";
      std::exit(1);
    }
  }
  //third part (ignored by until) (mainly for matrices)
  for(int i = outDim-until; i<outDim; i++ ){
      out1->shape.push_back(this->shape[i-thisOffset]);
      out2->shape.push_back(other.shape[i-otherOffset]);
  }

  out1->data = this->data;
  out2->data = other.data;
  // std::cout << "br finie\n";
}



// TODO: do try catch block for broadcasting,
// rn we only broadcast other so first arg will always be the final shape
Tensor Tensor::operator+(const Tensor &other) {

  Tensor* br1 = new Tensor();
  Tensor* br2 = new Tensor();
  this->broadcast(other, 0, br1, br2);
  Tensor out(br1->shape);
  iterate(this->shape, this->stride,
          [&](int offset, const std::vector<int> &x) {
            out.get(x) = br1->get(x) + br2->get(x);
          });
  delete br1;
  delete br2;
  return out;
}

Tensor Tensor::operator-(const Tensor &other) {

  Tensor* br1 = new Tensor();
  Tensor* br2 = new Tensor();
  this->broadcast(other, 0, br1, br2);
  Tensor out(br1->shape);
  iterate(this->shape, this->stride,
          [&](int offset, const std::vector<int> &x) {
            out.get(x) = br1->get(x) - br2->get(x);
          });
  delete br1;
  delete br2;
  return out;
}

Tensor Tensor::operator-() {
  Tensor out(this->shape);
  iterate(this->shape, this->stride,
          [&](int offset, const std::vector<int> &x) {
            out.get(x) = -this->get(x);
          });
  return out;
}
Tensor Tensor::operator+(const double other) {
  Tensor out(this->shape);
  iterate(this->shape, this->stride,
          [&](int offset, const std::vector<int> &x) {
            out.get(x) = this->get(x) + other;
          });
  return out;
}
Tensor Tensor::operator*(const double other) {
  Tensor out(this->shape);
  iterate(this->shape, this->stride,
          [&](int offset, const std::vector<int> &x) {
            out.get(x) = this->get(x) * other;
          });
  return out;
}
Tensor Tensor::operator*(const Tensor &other) {
  int other_dim = other.shape.size();

  Tensor* br1 = new Tensor();
  Tensor* br2 = new Tensor();
  this->broadcast(other, 0, br1, br2);
  Tensor out(br1->shape);
  iterate(this->shape, this->stride,
          [&](int offset, const std::vector<int> &x) {
            out.get(x) = br1->get(x) * br2->get(x);
          });
  delete br1;
  delete br2;
  return out;
}
// TODO implement matrix multiplication but for tensors (repeating matrix mult)
Tensor Tensor::mm(const Tensor &other) {
  assert(this->shape.size() >= 2 && other.shape.size() >= 2);
  
  auto start = std::chrono::steady_clock::now();

  int ed1 = this->effectiveDim();
  int ed2 = other.effectiveDim();
  Tensor* mat1_broadcasted = new Tensor(); 
  Tensor* mat2_broadcasted = new Tensor();

  this->broadcast(other, 2, mat1_broadcasted, mat2_broadcasted);
  int dim = mat2_broadcasted->shape.size();
  int k = mat1_broadcasted->shape[dim-1];
  assert(k == mat2_broadcasted->shape[dim - 2]);
  std::vector<int> newShape = mat2_broadcasted->shape;
  newShape[dim - 2] = mat1_broadcasted->shape[dim -2];
  Tensor out(newShape);
  
  int thisMovingStride = mat1_broadcasted->stride[dim-1]; 
  int otherMovingStride = mat2_broadcasted->stride[dim-2]; 
  iterate(out.shape, out.stride, [&](int offset, const std::vector<int> &x) {
    int thisPos =0 ,otherPos = 0;
    for(int i = 0; i<dim; i++){
      otherPos += x[i]*mat2_broadcasted->stride[i];
      thisPos += x[i]*mat1_broadcasted->stride[i];
    }

    thisPos -= x[dim-1]*thisMovingStride;
    otherPos -= x[dim-2]*otherMovingStride;
    for (int i = 0; i < k; i++) {

      (*out.data)[offset] += (*mat1_broadcasted->data)[thisPos + thisMovingStride*i] * (*mat2_broadcasted->data)[otherPos + otherMovingStride*i];
    }
  });
  delete mat1_broadcasted;
  delete mat2_broadcasted;

  auto end = std::chrono::steady_clock::now();
  auto duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(end- start);
  
  std::cout << "mm took  :" << duration.count() << '\n';
  return out;
}

Tensor Tensor::power(double expo) {
  Tensor out(this->shape);
  iterate(out.shape, out.stride, [&](int offset, const std::vector<int> &x) {
    out.get(x) = pow(this->get(x), expo);
  });
  return out;
}

Tensor Tensor::sigmoid() {
  Tensor out(this->shape);
  iterate(out.shape, out.stride, [&](int offset, const std::vector<int> &x) {
    out.get(x) = 1 / (1 + exp(-this->get(x)));
  });
  return out;
}

Tensor Tensor::sigmoidDeriv() {
  Tensor out(this->shape);
  iterate(out.shape, out.stride, [&](int offset, const std::vector<int> &x) {
    out.get(x) = exp(-this->get(x)) / pow(1 + exp(-this->get(x)), 2);
  });
  return out;
}

Tensor Tensor::sum() {
  double acc = 0;
  iterate(this->shape, this->stride,
          [&](int offset, const std::vector<int> &x) { acc += this->get(x); });
  Tensor out;
  out.get() = acc;
  return out;
}

Tensor Tensor::sum(std::vector<int> axes) {
  std::sort(axes.begin(),axes.end());
  std::vector<int> outShape;
  for(int v: outShape){
    outShape.push_back(this->shape[v]);
  }
  Tensor out(outShape);
  double acc = 0;
  iterate(this->shape, this->stride,
          [&](int offset, const std::vector<int> &x) {
            std::vector<int> currentIndex;
            for(int v: outShape){
              currentIndex.push_back(x[v]);
            }
            out.get(currentIndex) = this->get(x);


          });
  return out;
}
// not sure this makes a reference to data.
Tensor Tensor::transpose() {
  assert(this->shape.size() >= 2);
  std::vector<int> newShape = this->shape;
  int dim = this->shape.size();
  int n = newShape[dim-2];
  int m = newShape[dim-1];
  newShape[dim- 2] = m;
  newShape[dim- 1] = n;
  Tensor out(newShape);
  out.stride[dim-2] = (dim-2)?out.stride[dim-3]*out.shape[dim-3]:1; //c horrible
  out.stride[dim-1]= out.stride[dim-2]*m;
  out.data = this->data;
  return out;
}
void Tensor::printShape() {
  std::cout << "[";
  for (const int &i : this->shape) {
    std::cout << i << ", ";
  }
  std::cout << "]";
}

void Tensor::fillRandom() {
  std::mt19937_64 rng(std::random_device{}());
  std::uniform_real_distribution<double> dist(-0.1, 0.1);

  iterate(
      this->shape, this->stride,
      [&](int offset, const std::vector<int> &x) { this->get(x) = dist(rng); });
  std::cout << "\n";
}
