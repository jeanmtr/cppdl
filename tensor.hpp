#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>



//TODO: using templates
class Tensor{
   std::vector<double> data;
  public: 
     std::vector<int> shape;
     std::vector<int> stride;
     int size;
     Tensor(const std::vector<int>& shape);
     Tensor();
     double& get(const std::vector<int>& pos);
     const double& get(const std::vector<int>& pos) const;
     double& get();
     const double& get() const;
     //double get(vector<int> pos); idk
     Tensor operator+(const Tensor& other);
     Tensor operator*(const Tensor& other);
     Tensor mm(const Tensor& other);
     Tensor sigmoid();
     Tensor sigmoidDeriv();
     void printShape();
     Tensor transpose();
};


template <typename Func>
void iterate(const std::vector<int>& shape, const std::vector<int>& stride, Func&& fn);
