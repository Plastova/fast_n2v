#ifdef OPTIMIZED
#include "../snap/Snap.h"

struct Metadata {
  static const size_t CACHE_LINE_SIZE = 64;
};

template <class T>
class TRndBatch {
public:
  TRndBatch(const unsigned int seed, const int nstreams = 1, const int brng = 0): brng{brng}, seed{seed}, nstreams{nstreams}
  {
    IAssert(nstreams >= 1);
    MemPtrs = new T* [nstreams];
    StreamMemSize = new int [nstreams];
    streams = new TRnd [nstreams];
    for (int k = 0; k < nstreams; ++k) {
      TRnd Rnd(seed+k);
      streams[k] = Rnd;
      StreamMemSize[k] = 0;
    }
  }

  ~TRndBatch(){
    for (int k = 0; k < nstreams; ++k) {
      std::free(MemPtrs[k]);
    }
    if (MemPtrs != nullptr){delete[] MemPtrs;}
    if (StreamMemSize != nullptr){delete[] StreamMemSize;}
    if (streams != nullptr){delete[] streams;}
  }

  TRndBatch(const TRndBatch &Rnd) = delete;
  T* GetRndUniform(const int ValsCnt, const T a, const T b, const int StreamI = 0, const int method = 0);


private:
  T* GetMemPtr(int ValsCnt, int StreamI = 0);
  const int brng;
  const int nstreams;
  const unsigned int seed;
  T** MemPtrs = nullptr;
  int* StreamMemSize = nullptr;
  TRnd* streams = nullptr;
};
#endif
