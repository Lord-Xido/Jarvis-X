#!/usr/bin/env sh
set -eu
rm -rf build/test-classes
mkdir -p build/test-classes
find src/main/java src/test/java -name '*.java' | sort > build/test-sources.txt
javac --release 21 -encoding UTF-8 -d build/test-classes @build/test-sources.txt
java -ea -cp build/test-classes com.moagi.omega.KernelSelfTest
