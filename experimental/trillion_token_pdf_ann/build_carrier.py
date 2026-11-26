#!/usr/bin/env python3
"""Build a new PDF carrier and independently audit both reversible loops."""
import argparse
import json
from pathlib import Path
import fitz
import jarvis_trillion_ann as ann
import jarvis_pdf_bitvm as bit

EXPECTED = '2e81a7f2c992653af180e43a80cdfb02aafc71fc6a8a7df99313a8b2ef51228c'

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def build(dest):
    dest.mkdir(parents=True, exist_ok=True)
    a = ann.run(8, 1000, b'JARVIS-X')
    b = ann.run(8, 1000, b'JARVIS-X')
    require(a['trace'] == b['trace'], 'non-deterministic replay')
    require(a['token_stream_sha256'] == EXPECTED, 'token regression')
    require(a['emitted_tokens'] == 512000 and a['reference_errors'] == 0, 'ANN verification failed')
    report = {'architecture': ann.scale(), 'execution': a, 'replay_identical': True}
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((48, 56), 'JARVIS X - Trillion-token ANN architecture', fontsize=17)
    lines = [
        'Design target: 1000^3 logical streams x 1000 tokens/s = 10^12 tokens/s.',
        'Measured reference: 8^3 integer ANN, 1000 iterations, 512000 symbols.',
        'Untrained 32-symbol vocabulary; not a language model.',
        'ANN bytecode: ' + ann.PROGRAM.hex(' '),
        'Bit bytecode: ' + bit.PROGRAM.hex(' '),
        'Morton addressing, toroidal neighbors, encode, ANN, decode, verify, commit.',
        'Separate bit codec: XOR adjacent bits, pack, inverse XOR, SHA-256, commit.',
        'Target dense weight-matmul compute: 2e21 FLOP/s for 1B parameters.',
        'The target is not measured throughput. Execution requires an external VM.',
        'This carrier embeds source, both bytecode programs and the encoded spec.'
    ]
    page.insert_textbox(fitz.Rect(48, 85, 550, 740), '\n\n'.join(lines), fontsize=11)
    spec = doc.tobytes(garbage=4, deflate=True)
    reconstructed, packet, spec_audit = bit.run(spec, 1000)
    require(reconstructed == spec and bit.decode(packet) == spec, 'spec round trip failed')
    damaged = bytearray(packet)
    damaged[-1] ^= 1
    try:
        bit.decode(damaged)
    except ValueError:
        pass
    else:
        raise RuntimeError('corruption accepted')
    root = Path(__file__).resolve().parent
    for name in ('jarvis_trillion_ann.py', 'jarvis_pdf_bitvm.py', 'build_carrier.py'):
        doc.embfile_add(name, (root / name).read_bytes(), filename=name)
    for name, data in {
        'ann_program.bin': ann.PROGRAM,
        'bit_program.bin': bit.PROGRAM,
        'ann_trace.json': json.dumps(report, indent=2).encode(),
        'payload.jxb': packet,
        'payload_audit.json': json.dumps(spec_audit, indent=2).encode()
    }.items():
        doc.embfile_add(name, data, filename=name)
    carrier = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    with fitz.open(stream=carrier, filetype='pdf') as opened:
        require(bit.decode(opened.embfile_get('payload.jxb')) == spec, 'embedded decode failed')
    result, _, carrier_audit = bit.run(carrier, 1000)
    require(result == carrier, 'carrier round trip failed')
    (dest / 'jarvis_trillion_token_ann.pdf').write_bytes(carrier)
    (dest / 'reconstructed_spec.pdf').write_bytes(spec)
    (dest / 'ann_trace.json').write_text(json.dumps(report, indent=2) + '\n')
    (dest / 'pdf_audit.json').write_text(json.dumps({
        'embedded_specification': spec_audit, 'carrier': carrier_audit,
        'corruption_rejected': True
    }, indent=2) + '\n')
    print('PASS: 512000 toy symbols, deterministic replay, two 1000-round PDF audits')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    build(parser.parse_args().out)
