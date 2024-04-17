#ifndef WORD_2_VEC_H
#define WORD_2_VEC_H

#include "../snap/Snap.h"

//Learns embeddings using SGD, Skip-gram with negative sampling.
void LearnEmbeddings(TVVec<TInt, int64>& WalksVV, const int& Dimensions,
  const int& WinSize, const int& Iter, const bool& Verbose,
  TIntFltVH& EmbeddingsHV);

//Max x for e^x. Value taken from original word2vec code.
const int MaxExp = 6;

//Size of e^x precomputed table.
const int ExpTablePrecision = 10000;
const int TableSize = MaxExp*ExpTablePrecision*2;

//Number of negative samples. Value taken from original word2vec code.
const int NegSamN = 5;

//Learning rate for SGD. Value taken from original word2vec code.
const double StartAlpha = 0.025;

#ifdef OPTIMIZED

#include "batch_rnd.h"

#define OVERFLOW_CHECK_BY_MULTIPLICATION(type, op1, op2)                                          \
    {                                                                                             \
        if (!(0 == (op1)) && !(0 == (op2)))                                                       \
        {                                                                                         \
            volatile type r = (op1) * (op2);                                                      \
            r /= (op1);                                                                           \
            if (!(r == (op2))) throw std::runtime_error("Multiplication overflow");               \
        }                                                                                         \
    }


const double AlphaFact = 0.0001;
const int VerboseWordCnt = 10000;


template <typename FPType>
class TAlignedBuffer {
public:
  TAlignedBuffer(size_t Lines, size_t Columns) : Lines(Lines), Columns(Columns) {
    OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, Columns, sizeof(FPType));
    const size_t RawBytesPerLine = Columns * sizeof(FPType);
    const size_t CacheLinesPerLine = RawBytesPerLine / Metadata::CACHE_LINE_SIZE + !!(RawBytesPerLine % Metadata::CACHE_LINE_SIZE);
    OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, CacheLinesPerLine, Metadata::CACHE_LINE_SIZE);
    BytesPerLine = CacheLinesPerLine * Metadata::CACHE_LINE_SIZE;
    OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, BytesPerLine, Lines);
    const size_t RawFullBytes = BytesPerLine * Lines;
    const size_t CacheLinesFull = RawFullBytes / Metadata::CACHE_LINE_SIZE + !!(RawFullBytes % Metadata::CACHE_LINE_SIZE);
    OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, CacheLinesFull, Metadata::CACHE_LINE_SIZE);
    const size_t FullBytes = CacheLinesFull * Metadata::CACHE_LINE_SIZE;
    auto allocated = aligned_alloc(Metadata::CACHE_LINE_SIZE, FullBytes);
    if (allocated == nullptr) {
      throw std::bad_alloc();
    }
    DataStart = reinterpret_cast<char*>(allocated);
  }

  TAlignedBuffer(const TAlignedBuffer& rhs) = delete;
  TAlignedBuffer(TAlignedBuffer&& rhs) : Lines(rhs.Lines), Columns(rhs.Columns),
      BytesPerLine(rhs.BytesPerLine), DataStart(rhs.DataStart) {
    rhs.Lines = 0;
    rhs.Columns = 0;
    rhs.BytesPerLine = 0;
    rhs.DataStart = nullptr;
  }

  TAlignedBuffer& operator=(const TAlignedBuffer& rhs) = delete;
  TAlignedBuffer& operator=(TAlignedBuffer&& rhs) {
    if (this == &rhs) {
      return *this;
    }
    Lines = rhs.Lines;
    Columns = rhs.Columns;
    BytesPerLine = rhs.BytesPerLine;
    DataStart = rhs.DataStart;
    rhs.Lines = 0;
    rhs.Columns = 0;
    rhs.BytesPerLine = 0;
    rhs.DataStart = nullptr;
  }

  ~TAlignedBuffer() {
    free(DataStart);
  }

  size_t GetXDim() const {
    return Lines;
  }

  size_t GetYDim() const {
    return Columns;
  }

  FPType* GetLine(size_t LineIdx) {
    return reinterpret_cast<FPType*>(DataStart + BytesPerLine * LineIdx);
  }

private:
  size_t Lines{};
  size_t Columns{};
  size_t BytesPerLine{};
  char* DataStart{};
};
#endif //OPTIMIZED
#endif //WORD_2_VEC_H
