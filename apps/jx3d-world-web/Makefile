.PHONY: run test native
run:
	python3 server.py
test:
	python3 -m unittest discover -s tests -v
native:
	cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j
