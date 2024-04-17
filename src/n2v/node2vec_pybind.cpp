#include "node2vec.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
PYBIND11_MODULE(n2vcpp_opt, m) {
    py::class_<PWNet>(m, "PWNet");
    py::class_<TIntFltVH>(m, "TIntFltVH");
    py::class_<TVVec<TInt,int64>>(m, "TVVec<TInt,int64>");

    m.def("ReadGraph",
    [](std::string &input_file, bool directed, bool weighted, bool verbose){
      PWNet InNet = PWNet::New();
      TStr InFile(input_file.c_str());
      ReadGraph(InFile, directed, weighted, verbose, InNet);
      return InNet;
    });

    m.def("node2vec",
    [](PWNet& InNet, const double& ParamP, const double& ParamQ,
        const int& Dimensions, const int& WalkLen, const int& NumWalks,
        const int& WinSize, const int& Iter, const bool& Verbose,
        const bool& OutputWalks){
      TIntFltVH EmbeddingsHV;
      TVVec <TInt, int64> WalksVV;
      node2vec(InNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize, Iter,
        Verbose, OutputWalks, WalksVV, EmbeddingsHV);
        py::tuple tup = py::make_tuple(EmbeddingsHV, WalksVV);
        return tup;
    });

    m.def("WriteOutput",
    [](std::string &output_file, TIntFltVH& EmbeddingsHV, TVVec<TInt, int64>& WalksVV,
        bool& OutputWalks) {
        TStr OutFile(output_file.c_str());
        WriteOutput(OutFile, EmbeddingsHV, WalksVV, OutputWalks);
    });
};

PYBIND11_MODULE(n2vcpp_ref, m) {
    m.def("ReadGraph",
    [](std::string &input_file, bool directed, bool weighted, bool verbose){
      PWNet InNet = PWNet::New();
      TStr InFile(input_file.c_str());
      ReadGraph(InFile, directed, weighted, verbose, InNet);
      return InNet;
    });

    m.def("node2vec",
    [](PWNet& InNet, const double& ParamP, const double& ParamQ,
        const int& Dimensions, const int& WalkLen, const int& NumWalks,
        const int& WinSize, const int& Iter, const bool& Verbose,
        const bool& OutputWalks){
      TIntFltVH EmbeddingsHV;
      TVVec <TInt, int64> WalksVV;
      node2vec(InNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize, Iter,
        Verbose, OutputWalks, WalksVV, EmbeddingsHV);
        py::tuple tup = py::make_tuple(EmbeddingsHV, WalksVV);
        return tup;
    });

    m.def("WriteOutput",
    [](std::string &output_file, TIntFltVH& EmbeddingsHV, TVVec<TInt, int64>& WalksVV,
        bool& OutputWalks) {
        TStr OutFile(output_file.c_str());
        WriteOutput(OutFile, EmbeddingsHV, WalksVV, OutputWalks);
    });
};
