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
     //double get(vector<int> pos); idk
     Tensor operator+(const Tensor& other);
     Tensor operator*(const Tensor& other);

};
