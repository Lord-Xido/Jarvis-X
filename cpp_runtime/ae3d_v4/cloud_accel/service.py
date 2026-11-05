"""Authenticated bounded HTTP service for JX AE3D v4-compatible batched inference.

Run: JX_MODEL_PATH=... JX_API_KEY=... uvicorn service:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import os
import secrets
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

from jx_accel import Archive, NeuralCodec, Volume, read_bounded

MAX_REQUEST_BYTES = 16*1024*1024


def create_app(model_path: str, api_key: str, device: str = 'cpu', batch_blocks: int = 4096,
               max_body: int = MAX_REQUEST_BYTES) -> FastAPI:
    if len(api_key) < 16:
        raise ValueError('API key must have at least 16 characters')
    if not 1024 <= max_body <= 64*1024*1024:
        raise ValueError('invalid body limit')
    engine = NeuralCodec(read_bounded(model_path), device=device, batch_blocks=batch_blocks)
    api = FastAPI(title='JARVIS-X AE3D Cloud v5', version='5.0.0', docs_url=None, redoc_url=None)

    def auth(authorization: Annotated[str | None, Header()] = None) -> None:
        if authorization is None or not secrets.compare_digest(authorization, f'Bearer {api_key}'):
            raise HTTPException(status_code=401, detail='unauthorized',headers={'WWW-Authenticate':'Bearer'})

    async def read_request(req: Request) -> bytes:
        length = req.headers.get('content-length')
        if length and (not length.isdigit() or int(length) > max_body):
            raise HTTPException(status_code=413,detail='payload exceeds configured limit')
        data = bytearray()
        async for chunk in req.stream():
            if len(data)+len(chunk)>max_body:
                raise HTTPException(status_code=413,detail='payload exceeds configured limit')
            data.extend(chunk)
        return bytes(data)

    @api.exception_handler(ValueError)
    async def reject_input(_req: Request, exc: ValueError):
        return JSONResponse(status_code=422,content={'detail':str(exc)})

    @api.get('/healthz')
    def health():
        return {'status':'ok','model_sha256':engine.sha256,'device':engine.device,
                'model_parameters':5004,'precision':'float64','batch_blocks':engine.batch_blocks}

    @api.post('/v1/encode',dependencies=[Depends(auth)])
    async def encode(req: Request, exact: bool = Query(default=False),
                     ox: int = Query(default=0,ge=0,lt=10**12),
                     oy: int = Query(default=0,ge=0,lt=10**12),
                     oz: int = Query(default=0,ge=0,lt=10**12)):
        vol = Volume.parse(await read_request(req))
        packet=engine.encode(vol,exact,(ox,oy,oz)).serialize()
        return Response(packet,media_type='application/octet-stream',headers={'X-JX-Exact':str(int(exact))})

    @api.post('/v1/decode',dependencies=[Depends(auth)])
    async def decode(req: Request):
        arc=Archive.parse(await read_request(req))
        vol=engine.decode(arc)
        return Response(vol.serialize(),media_type='application/octet-stream')

    @api.post('/v1/query',dependencies=[Depends(auth)])
    async def query(req: Request,
                    x: int = Query(ge=0,lt=10**12),
                    y: int = Query(ge=0,lt=10**12),
                    z: int = Query(ge=0,lt=10**12)):
        arc=Archive.parse(await read_request(req))
        return engine.probe(arc,(x,y,z))

    return api


def app_from_env() -> FastAPI:
    key=os.environ.get('JX_API_KEY','')
    model=os.environ.get('JX_MODEL_PATH','')
    if not model:
        raise ValueError('set JX_MODEL_PATH to an existing v4 model checkpoint')
    return create_app(model, key, device=os.environ.get('JX_ACCEL_DEVICE','cpu'),
                      batch_blocks=int(os.environ.get('JX_BATCH_BLOCKS','4096')))


# Resolve model and fail closed at startup, not at the first network request.
# Environment defaults are omitted to prevent accidental exposure of an unprotected API.
app = app_from_env() if os.environ.get('JX_MODEL_PATH') else None
