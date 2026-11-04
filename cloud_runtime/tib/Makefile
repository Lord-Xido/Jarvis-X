CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
.PHONY: all test clean bench
all: jx3d_1tib jx3d_media
jx3d_1tib: jx3d_1tib.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
jx3d_media: jx3d_media_stream.cpp jx3d_1tib.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
test: all
	python3 test_engine.py
bench: all
	./jx3d_1tib --tiles 1024 --corrupt-test
clean:
	rm -f jx3d_1tib jx3d_media
