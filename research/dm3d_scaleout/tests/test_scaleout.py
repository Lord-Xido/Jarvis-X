import numpy as np
import pytest
from dm3d_multimodal.data import SceneDataset
from dm3d_scale.shards import write_shards,MMapSceneDataset
from dm3d_scale.index import SegmentIndex
from dm3d_scale.tiles import windows,crop_pad,stitch,estimate_virtual


def test_sharded_roundtrip(tmp_path):
    source=SceneDataset(count=7,start=20)
    report=write_shards(source,tmp_path/'train',shard_size=3)
    assert report['total']==7 and len(report['shards'])==3
    dataset=MMapSceneDataset(tmp_path/'train',cache_shards=1)
    assert len(dataset)==7
    for i in [6,1,3,0,6,5]:
        actual=dataset[i]
        expected=source[i]
        for key in ('image','video','audio','text'):
            assert np.array_equal(actual[key].numpy(),expected[key].numpy())
        assert int(actual['sample_id'])==expected['sample_id']
    with pytest.raises(IndexError):
        _=dataset[7]
    with pytest.raises(FileExistsError):
        write_shards(source,tmp_path/'train',shard_size=3)


def test_retrieval_segments_are_exact(tmp_path):
    idx=SegmentIndex(tmp_path/'vectors',dimension=4)
    idx.append([0,1],[[1,0,0,0],[0,1,0,0]])
    idx.append([2,3],[[.9,.1,0,0],[0,0,1,0]])
    reopened=SegmentIndex(tmp_path/'vectors',dimension=4)
    answers=reopened.topk(np.array([1,0,0,0]),k=3,block=1)
    assert [v['id'] for v in answers]==[0,2,1]
    assert abs(answers[0]['cosine']-1)<1e-7
    assert np.allclose(reopened.embedding(2),[.9,.1,0,0],atol=1e-7)
    assert reopened.topk([1,0,0,0],k=1,exclude=0)[0]['id']==2
    with pytest.raises(ValueError):
        reopened.append([0],[[0,1,0,0]])
    with pytest.raises(ValueError):
        reopened.append([5],[[float('nan'),0,0,0]])


def test_virtual_tile_stitch():
    original=np.arange(3*23*29,dtype='float32').reshape(3,23,29)
    parts=[]
    for y,x in windows(23,29,16,4):
        parts.append((y,x,crop_pad(original,y,x,16)))
    reconstructed=stitch(parts,23,29,16)
    assert np.array_equal(reconstructed,original)
    footprint=estimate_virtual(8000000,8000000)
    assert footprint['tiles_x']==3907
    assert footprint['virtual_tiles']==3907**2
    assert footprint['max_resident_bytes']==122*2048**2*16


def test_invalid_inputs(tmp_path):
    with pytest.raises(ValueError):
        list(windows(4,4,16,16))
    with pytest.raises(ValueError):
        SegmentIndex(tmp_path,dimension=-1)
    with pytest.raises(ValueError):
        stitch([],16,16)


def test_huge_coordinates_are_lazy():
    from itertools import islice
    first=list(islice(windows(8_000_000,8_000_000,tile=2048),4))
    assert first==[(0,0),(0,2048),(0,4096),(0,6144)]


def test_exact_validation_partition(tmp_path):
    from dm3d_scale.fsdp2 import PaddedValidation
    source=SceneDataset(count=7,start=0)
    parts=[PaddedValidation(source,rank,3) for rank in range(3)]
    assert [len(part) for part in parts]==[3,3,3]
    ids=[]
    for part in parts:
        ids += [int(part[i]['sample_id']) for i in range(len(part)) if bool(part[i]['valid'])]
    assert sorted(ids)==list(range(7))
