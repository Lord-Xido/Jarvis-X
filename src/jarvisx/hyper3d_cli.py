from __future__ import annotations

import argparse
from typing import Optional, Sequence
import uvicorn

def parser():
    p=argparse.ArgumentParser(prog="jarvisx-hyper3d",description="Launch Jarvis-X Hyper3D multimodal programming interface")
    p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=8899); p.add_argument("--reload",action="store_true")
    return p

def main(argv:Optional[Sequence[str]]=None):
    a=parser().parse_args(argv); uvicorn.run("jarvisx.hyper3d_api:app",host=a.host,port=a.port,reload=a.reload); return 0

if __name__=="__main__": raise SystemExit(main())
