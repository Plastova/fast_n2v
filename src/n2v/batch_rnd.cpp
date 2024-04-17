#ifdef OPTIMIZED
#include "batch_rnd.h"

template <class T>
T* TRndBatch<T>::GetRndUniform(const int ValsCnt, const T a, const T b, const int StreamI, const int method) {
    T* RndVals = GetMemPtr(ValsCnt, StreamI);
    if constexpr(std::is_same_v<T, double>) {
      auto Rnd = streams[StreamI];
      for (int i=0; i<ValsCnt; ++i) {
        RndVals[i] = Rnd.GetUniDev();
      }
    }
    if constexpr(std::is_same_v<T, int>) {
      auto Rnd = streams[StreamI];
      for (int i=0; i<ValsCnt; ++i) {
        RndVals[i] = Rnd.GetUniDevInt();
      }
    }
    return RndVals;
  }

template <class T>
T* TRndBatch<T>::GetMemPtr(int ValsCnt, int StreamI) {
    IAssert(StreamI >= 0 && StreamI < nstreams);
    if (ValsCnt <= 0) {
          EFailR("ValsCnt must be greater than zero.");
    }
    if (ValsCnt > StreamMemSize[StreamI] && StreamMemSize[StreamI] > 0) {
      std::free(MemPtrs[StreamI]);
      MemPtrs[StreamI] = static_cast<T*>(std::malloc(ValsCnt * sizeof(T)));
      if (MemPtrs[StreamI] == NULL) {
        EFailR("Error memory allocation in random generation.");
      }
      StreamMemSize[StreamI] = ValsCnt;
      return MemPtrs[StreamI];
    }
    if (StreamMemSize[StreamI] == 0) {
      MemPtrs[StreamI] = static_cast<T*>(std::malloc(ValsCnt * sizeof(T)));
      if (MemPtrs[StreamI] == NULL) {
        EFailR("Error memory allocation in random generation.");
      }
      StreamMemSize[StreamI] = ValsCnt;
    }
    return MemPtrs[StreamI];
  }

template class TRndBatch<double>;
template class TRndBatch<float>;
template class TRndBatch<int>;

#endif
