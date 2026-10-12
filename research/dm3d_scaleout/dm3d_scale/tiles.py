"""Virtual image and video tiling: never allocate the global canvas."""
from __future__ import annotations
import math
import numpy as np

def windows(height,width,tile=16,overlap=0):
    if height<1 or width<1 or tile<1 or not 0<=overlap<tile:
        raise ValueError('invalid tiling geometry')
    step=tile-overlap
    ny=math.ceil(max(height-tile,0)/step)+1
    nx=math.ceil(max(width-tile,0)/step)+1
    # Generator, not an eagerly allocated grid.
    for row in range(ny):
        y=min(row*step,max(0,height-tile))
        for col in range(nx):
            x=min(col*step,max(0,width-tile))
            yield y,x

def crop_pad(image, y, x, tile=16):
    """Image shape [...,H,W] with spatial padding on right/bottom only."""
    if image.ndim<2 or tile<1:
        raise ValueError('expected [...,H,W]')
    patch=image[...,y:y+tile,x:x+tile]
    return np.pad(patch,[(0,0)]*(patch.ndim-2)+[(0,tile-patch.shape[-2]),(0,tile-patch.shape[-1])])

def stitch(patches,height,width,tile=16):
    """Overlap-average [y,x,output] tiles; one target canvas at requested finite size."""
    out=None; weights=np.zeros((height,width),dtype=np.float32)
    for y,x,p in patches:
        p=np.asarray(p,dtype=np.float32)
        if p.shape[-2:]!=(tile,tile) or y<0 or x<0:
            raise ValueError('invalid returned patch')
        if out is None:
            out=np.zeros((*p.shape[:-2],height,width),dtype=np.float32)
        h=min(tile,height-y);w=min(tile,width-x)
        if h<=0 or w<=0:
            raise ValueError('patch outside image')
        out[...,y:y+h,x:x+w]+=p[...,:h,:w]
        weights[y:y+h,x:x+w]+=1
    if out is None or not np.all(weights):
        raise ValueError('coverage incomplete')
    return out/weights

def estimate_virtual(height,width,tile=2048,channels=4,bytes_per_channel=4,active_slots=122):
    if min(height,width,tile,channels,bytes_per_channel,active_slots)<1:
        raise ValueError('invalid geometry')
    nx=math.ceil(width/tile);ny=math.ceil(height/tile)
    each=tile*tile*channels*bytes_per_channel
    return {'tiles_x':nx,'tiles_y':ny,'virtual_tiles':nx*ny,
            'per_tile_bytes':each,'max_resident_bytes':min(nx*ny,active_slots)*each,
            'dense_virtual_bytes':height*width*channels*bytes_per_channel}
