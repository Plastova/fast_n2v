#include "node2vec.h"

int main(int argc, char* argv[]) {
  TStr InFile,OutFile;
  int Dimensions, WalkLen, NumWalks, WinSize, Iter;
  double ParamP, ParamQ;
  bool Directed, Weighted, Verbose, OutputWalks;
  ParseArgs(argc, argv, InFile, OutFile, Dimensions, WalkLen, NumWalks, WinSize,
   Iter, Verbose, ParamP, ParamQ, Directed, Weighted, OutputWalks);
  PWNet InNet = PWNet::New();
  TIntFltVH EmbeddingsHV;
  TVVec <TInt, int64> WalksVV;
  ReadGraph(InFile, Directed, Weighted, Verbose, InNet);
  node2vec(InNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize, Iter,
   Verbose, OutputWalks, WalksVV, EmbeddingsHV);
  WriteOutput(OutFile, EmbeddingsHV, WalksVV, OutputWalks);
  return 0;
}
