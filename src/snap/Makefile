include ../../Makefile.config

all: Snap.o

Snap.o: *.h *.cpp
	$(CC) -c $(CXXFLAGS) -fPIC Snap.cpp

clean:
	rm -f *.o
