# Jarvis-X

Jarvis-X is a deterministic, auditable virtual machine with a reflex
control layer and policy gate.

## Install
```bash
git clone https://github.com/Lord-Xido/Jarvis-X.git
cd Jarvis-X
pip install -r requirements.txt
pip install .
```

## Moagi Ω Browser V2

The repository now includes an engine-independent Java 21 browser kernel under
[`omega-browser-v2/`](omega-browser-v2/). It provides transactional browser
commands, an origin-bound capability broker, semantic scene snapshots, frame
surface contracts, process supervision, and replaceable engine-adapter
boundaries for Chromium/JCEF and Servo.

```bash
cd omega-browser-v2
./test.sh
./build.sh
java -jar dist/moagi-omega-browser-v2.jar
```

The bundled engine is deterministic and intended for kernel verification. Real
web compatibility remains an adapter milestone rather than a claimed feature.
