/*
sudo apt install libomp-dev
apt show libomp-dev

g++  spm_lambda.cpp -o secuencial.out
g++ -fopenmp spm_lambda.cpp -o paralelo.out

*/
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>
#include <ctime>
#include <time.h>
#include <parallel/algorithm>
#include "include/aqsort.h"

int main()
{
    unsigned t0, t1;
    int num, c;
    std::vector<double> vals;
    srand(time(NULL));
    long long int sz = 100;
    
    for(c = 0; c < sz; c++)
    {
        num = 1 + rand() % (100 - 1);
        vals.push_back(num);
    }
    

    /*for (int i = 0; i < vals.size(); i++) 
        std::cout << vals[i] << ", ";
    std::cout << std::endl;*/

    auto comp = [&vals] (std::size_t i, std::size_t j) /* -> bool */ {
        if (vals[i] < vals[j])
            return true;        
        return false;
    };

    auto swap = [&vals] (std::size_t i, std::size_t j) {        
        std::swap(vals[i], vals[j]);
    };

    t0 = clock();
    //__gnu_parallel::sort( vals.begin(), vals.end(),  __gnu_parallel::quicksort_tag());
    aqsort::sort(vals.size(), &comp, &swap);
    //std::sort( vals.begin(), vals.end());
    t1 = clock();

   /* for (int i = 0; i < vals.size(); i++) 
        std::cout << vals[i] << ", ";
    std::cout << std::endl;*/

     
    double time = (double(t1-t0)/CLOCKS_PER_SEC);
    std::cout << "Execution Time: " << time << std::endl;
}
