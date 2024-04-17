import n2v_ext as n2v


input_path = "example/graph/graph.edgelist"
output_path = "example/emb/result_ref_p.emb"
graph = n2v.read_graph(input_path, directed=False, weighted=False, verbose=False)

params = {
    "dimension": 128,
    "walk_len": 80,
    "walks_per_src": 10,
    "context_size": 10,
    "num_epochs": 1,
    "return_param": 1.0,
    "inout_param": 1.0,
    "output_walks": False,
}

# Reference implementation
n2v_model_ref = n2v.N2VReference(**params)
n2v_model_ref.compute(graph, verbose=False)
n2v_model_ref.write_output(output_path)

input_path = "example/graph/graph.edgelist"
output_path = "example/emb/result_opt_p.emb"

# Optimized implementation
n2v_model_opt = n2v.N2VOptimized(**params)
n2v_model_opt.compute(graph, verbose=False)
n2v_model_opt.write_output(output_path)
