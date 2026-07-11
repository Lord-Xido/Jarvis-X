@echo off
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
mkdir build\classes
mkdir dist
dir /s /b src\main\java\*.java > build\sources.txt
javac --release 21 -encoding UTF-8 -d build\classes @build\sources.txt
jar --create --file dist\moagi-omega-browser-v2.jar --main-class com.moagi.omega.Main -C build\classes .
