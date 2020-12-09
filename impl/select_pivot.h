#ifndef AQSORT_IMPL_SELECT_PIVOT_H
#define AQSORT_IMPL_SELECT_PIVOT_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cassert>
#include <algorithm>
#include <vector>
#include <cstddef>

namespace aqsort
{
    namespace impl
    {
        template<typename Comp>
        inline std::size_t median(std::size_t i1, std::size_t i2, std::size_t i3, Comp* const comp)
        {
            if ((*comp)(i1, i3)) {
                if ((*comp)(i2, i1)) 
                    return i1;
                else if ((*comp)(i2, i3))
                    return i2;
                return i3;
            }

            if ((*comp)(i1, i2))
                return i1;
            else if ((*comp)(i3, i2))
                return i2;
            return i3;
        }

        // median-of-3 pivot
        template<typename Comp>
        inline std::size_t select_pivot_mo3(std::size_t start, std::size_t n, Comp* const comp)
        {
            // avoid unnecessary calling performance penalty
            assert (n > 0);

            std::size_t left   = start;
            std::size_t middle = start + (n >> 1);
            std::size_t right  = start + n - 1;

            return median(left, middle, right, comp);
        }        

        // median-of-medians pivot
        template<typename Comp>
        inline std::size_t select_pivot_mom(std::size_t start, std::size_t n, Comp* const comp)
        {
            // avoid unnecessary calling performance penalty
            assert (n > 0);

            std::size_t third1 =     n / 3;
            std::size_t third2 = 2 * n / 3;

            std::size_t i1 = start;
            std::size_t i2 = start + (third1 >> 1);
            std::size_t i3 = start + third1;
            std::size_t median1 = median(i1, i2, i3, comp);

            i1 = start + third1 + 1;
            i2 = start + third1 + ((third2 - third1) >> 1);
            i3 = start + third2;
            std::size_t median2 = median(i1, i2, i3, comp);

            i1 = start + third2 + 1;
            i2 = start + third2 + ((n - third2) >> 1);
            i3 = start + n - 1;
            std::size_t median3 = median(i1, i2, i3, comp);
            
            return median(median1, median2, median3, comp);
        }

        // median-of-5 pivot
        template<typename Comp>
        inline std::size_t parallel_mo5(std::size_t start, std::size_t n, std::vector<std::size_t> &medians, Comp* const comp)
        {
            // avoid unnecessary calling performance penalty
            assert (n > 0);

            const std::size_t p = omp_get_thread_num();
            start = start + (p * n);

            std::size_t a   = start;
            std::size_t b = start + (n >> 1);
            std::size_t c  = start + n - 1;
            std::size_t d = (a + c) / 4;
            std::size_t e  = 3 * (a + c) / 4;
            std::size_t tmp;

            if((*comp)(b, a)){
                tmp = a; a = b; b = tmp;
            }

            if((*comp)(d, c)){
                tmp = c; c = d; d = tmp;
            }

            if((*comp)(c, a)){
                tmp = b; b = d; d = tmp; 
                c = a;
            }

            a = e;

            if((*comp)(b, a)){
                tmp = a; a = b; b = tmp;
            }

            if((*comp)(a, c)){
                tmp = b; b = d; d = tmp; 
                a = c;
            }

            if((*comp)(d, a))
                return d;
            else
                return a;
        }

        // median-of-medians pivot
        template<typename Comp>
        inline void parallel_mom(std::size_t start, std::size_t n, std::size_t third1, std::size_t third2, std::vector<std::size_t> &medians, Comp* const comp)
        {
            // avoid unnecessary calling performance penalty
            assert (n > 0);

            const std::size_t p = omp_get_thread_num();
            std::size_t my_start = start + (p * n);

            std::size_t i1 = my_start;
            std::size_t i2 = my_start + (third1 >> 1);
            std::size_t i3 = my_start + third1;
            std::size_t median1 = median(i1, i2, i3, comp);

            i1 = my_start + third1 + 1;
            i2 = my_start + third1 + ((third2 - third1) >> 1);
            i3 = my_start + third2;
            std::size_t median2 = median(i1, i2, i3, comp);

            i1 = my_start + third2 + 1;
            i2 = my_start + third2 + ((n - third2) >> 1);
            i3 = my_start + n - 1;
            std::size_t median3 = median(i1, i2, i3, comp);
            
            medians[p] = median(median1, median2, median3, comp);
        }

        // median-of-medians pivot
        template<typename Comp>
        inline std::size_t select_pivot_pmom(std::size_t start, std::size_t n, std::size_t P, Comp* const comp)
        {
            // avoid unnecessary calling performance penalty
            assert (n > 0);

            const std::size_t offset = n / P;
            const std::size_t third1 =     offset / 3;
            const std::size_t third2 = 2 * offset / 3;
            std::vector<std::size_t> medians(P);

#pragma omp parallel num_threads(P)
            parallel_mom(start, offset, third1, third2, medians, comp);

            std::sort(medians.begin(), medians.end(), *comp);

            return medians[P / 2];
        }

        // median-of-medians pivot
        template<typename Comp>
        inline std::size_t select_pivot_pmo5(std::size_t start, std::size_t n, std::size_t P, Comp* const comp)
        {
            // avoid unnecessary calling performance penalty
            assert (n > 0);

            const std::size_t offset = n / P;
            std::vector<std::size_t> medians(P);

#pragma omp parallel num_threads(P)
            parallel_mo5(start, offset, medians, comp);

            std::sort(medians.begin(), medians.end(), *comp);
            return medians[P / 2];
        }
    }
}

#endif
