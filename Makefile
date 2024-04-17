include Makefile.config

SNAPDEPS = src/snap
N2VDIR = src/n2v
DEPH = $(N2VDIR)/n2v.h $(N2VDIR)/node2vec.h $(N2VDIR)/word2vec.h $(N2VDIR)/biasedrandomwalk.h
DEPCPP = $(N2VDIR)/n2v.cpp $(N2VDIR)/word2vec.cpp $(N2VDIR)/biasedrandomwalk.cpp

all: wheel

wheel: Snap.o
	python setup.py bdist_wheel

node2vec_optimized: $(N2VDIR)/node2vec.cpp $(DEPH) $(N2VDIR)/batch_rnd.h $(DEPCPP) $(N2VDIR)/batch_rnd.cpp Snap.o
	$(CC) $(CXXFLAGS) -DOPTIMIZED -o node2vec_opt $(N2VDIR)/node2vec.cpp $(N2VDIR)/batch_rnd.cpp $(DEPCPP) $(SNAPDEPS)/Snap.o -I$(SNAPDEPS) -I/$(N2VDIR) $(LIBS)

node2vec_reference: $(N2VDIR)/node2vec.cpp $(DEPH) $(DEPCPP) Snap.o
	$(CC) $(CXXFLAGS) -o node2vec_ref $(N2VDIR)/node2vec.cpp $(DEPCPP) $(SNAPDEPS)/Snap.o -I$(SNAPDEPS) -I/$(N2VDIR) $(LIBS)

Snap.o:
	make -C $(SNAPDEPS)

clean:
	$(MAKE) clean -C src/snap
	rm -f node2vec_opt node2vec_ref
	rm -rf build dist n2v_ext.egg-info
