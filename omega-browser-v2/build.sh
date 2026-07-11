#!/usr/bin/env sh
set -eu
rm -rf build dist
mkdir -p build/classes dist
find src/main/java -name '*.java' | sort > build/sources.txt
javac --release 21 -encoding UTF-8 -d build/classes @build/sources.txt
jar --create --file dist/moagi-omega-browser-v2.jar \
    --main-class com.moagi.omega.Main -C build/classes .
echo "Built dist/moagi-omega-browser-v2.jar"
