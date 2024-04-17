import n2vcpp_opt as n2v_opt
import n2vcpp_ref as n2v_ref


def read_graph(path, directed=False, weighted=False, verbose=False):
    graph = n2v_opt.ReadGraph(path, directed, weighted, verbose)
    return graph


class N2VBase:
    def __init__(
        self,
        dimension,
        walk_len,
        walks_per_src,
        context_size,
        num_epochs,
        return_param,
        inout_param,
        output_walks,
    ):
        self.dimension = dimension
        self.walk_len = walk_len
        self.walks_per_src = walks_per_src
        self.context_size = context_size
        self.num_epochs = num_epochs
        self.return_param = return_param
        self.inout_param = inout_param
        self.output_walks = output_walks
        self.res_tup = ()

    def compute(self, graph, verbose):
        self.res_tup = self.base_impl.node2vec(
            graph,
            self.return_param,
            self.inout_param,
            self.dimension,
            self.walk_len,
            self.walks_per_src,
            self.context_size,
            self.num_epochs,
            verbose,
            self.output_walks,
        )

    def write_output(self, output_path):
        self.base_impl.WriteOutput(
            output_path, self.res_tup[0], self.res_tup[1], self.output_walks
        )


class N2VOptimized(N2VBase):
    def __init__(
        self,
        dimension=128,
        walk_len=80,
        walks_per_src=10,
        context_size=10,
        num_epochs=1,
        return_param=1.0,
        inout_param=1.0,
        output_walks=False,
    ):
        super().__init__(
            dimension=dimension,
            walk_len=walk_len,
            walks_per_src=walks_per_src,
            context_size=context_size,
            num_epochs=num_epochs,
            return_param=return_param,
            inout_param=inout_param,
            output_walks=output_walks,
        )
        self.base_impl = n2v_opt


class N2VReference(N2VBase):
    def __init__(
        self,
        dimension=128,
        walk_len=80,
        walks_per_src=10,
        context_size=10,
        num_epochs=1,
        return_param=1.0,
        inout_param=1.0,
        output_walks=False,
    ):
        super().__init__(
            dimension=dimension,
            walk_len=walk_len,
            walks_per_src=walks_per_src,
            context_size=context_size,
            num_epochs=num_epochs,
            return_param=return_param,
            inout_param=inout_param,
            output_walks=output_walks,
        )
        self.base_impl = n2v_ref
