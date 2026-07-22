#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
#include <memory>
#include <random>


//TODO: using templates
class Tensor{
   std::shared_ptr<std::vector<double>> data;
  public: 
     std::vector<int> shape;
     std::vector<int> stride;
     int dim;
     Tensor(const std::vector<int>& shape);
     Tensor();
     double& get(const std::vector<int>& pos);
     const double& get(const std::vector<int>& pos) const;
     double& get();
     const double& get() const;
     int effectiveDim() const;
     //double get(vector<int> pos); idk
     Tensor operator+(const Tensor& other);
     Tensor operator+(const double other);
     Tensor operator*(const Tensor& other);
     Tensor operator*(const double other);
     Tensor operator-(const Tensor& other);
     Tensor operator-();
     Tensor sum();
     Tensor mm(const Tensor& other);
     Tensor sigmoid();
     Tensor power(double exp);
     Tensor sigmoidDeriv();
     void printShape();
     Tensor transpose();
     void broadcast(const Tensor& other, int until, Tensor* out1, Tensor* out2) const;
     void fillRandom();
};


template <typename Func>
void iterate(const std::vector<int>& shape, const std::vector<int>& stride, Func&& fn);
