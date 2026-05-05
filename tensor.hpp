#include <iostream>
#include <vector>
#include <assert.h>



//TODO: using templates
class Tensor{
   std::vector<double> data;
  public: 
     std::vector<int> shape;
     std::vector<int> stride;
     int size;
     Tensor(const std::vector<int>& shape);
     double& get(const std::vector<int>& pos);
     const double& get(const std::vector<int>& pos) const;
     //double get(vector<int> pos); idk
     Tensor operator+(const Tensor& other);
     Tensor operator*(const Tensor& other);

};


template <typename Func>
void iterate(const std::vector<int>& shape, const std::vector<int>& stride, Func&& fn);
