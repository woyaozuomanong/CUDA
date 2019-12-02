.SUFFIXES:.cu.o

CXX=nvcc
#CXXFLAGS= -std=c++11
SRCS=SimpleDivergence.cu
OBJS=$(SRCS:.cu=.o)
EXEC=SimpleDivergence

$(EXEC):$(OBJS)
	$(CXX) -o $(EXEC) $(OBJS)

$(OBJS):$(SRCS)
	$(CXX) -c $(SRCS)

	@echo "------echo------OK-----------"

#.cpp.o:
#	$(CXX) $(CXXFLAGS) -Wall -o $@ -c $<


clean:
	rm -rf $(EXEC) $(OBJS)
