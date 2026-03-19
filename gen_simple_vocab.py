#!/usr/bin/env python3
"""
gen_simple_vocab.py
A setup script added in this fork to generate a simple binary vocabulary 
format compatible with the metal_infer C engine, handling BPE byte-level decoding.
"""
import json
import struct
import sys

def get_bpe_mapping():
    # GPT-2 / Qwen byte-to-unicode mapping
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    # Map from char to byte
    return {chr(c): b for c, b in zip(cs, bs)}

def decode_bpe_string(s, mapping):
    try:
        # Map characters to bytes, then decode as UTF-8
        b = bytes([mapping[c] for c in s])
        return b
    except KeyError:
        # If character not in mapping, stick to regular encoding (it might be a special token)
        return s.encode('utf-8')

def main():
    tok_path = sys.argv[1] # tokenizer.json
    out_path = sys.argv[2] # vocab.bin

    with open(tok_path, 'r', encoding='utf-8') as f:
        t = json.load(f)

    # 1. Start with base vocab
    vocab = t['model']['vocab']
    # 2. Merge added tokens (special tokens)
    # added_tokens is a list of dicts: {"id": ..., "content": ...}
    for tok in t.get('added_tokens', []):
        vocab[tok['content']] = tok['id']

    # Sort by token_id to handle gaps if any (though usually contiguous)
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    mapping = get_bpe_mapping()
    
    # We need to ensure we have every ID from 0 to max_id
    max_id = sorted_vocab[-1][1]
    id_to_str = {idx: "<unk>" for idx in range(max_id + 1)}
    for s, idx in sorted_vocab:
        id_to_str[idx] = s

    with open(out_path, 'wb') as f:
        # Header: num_entries, max_id
        f.write(struct.pack('<I', max_id + 1))
        f.write(struct.pack('<I', max_id))
        
        for i in range(max_id + 1):
            s = id_to_str[i]
            # Decode the BPE string back to actual UTF-8 bytes
            b = decode_bpe_string(s, mapping)
            
            f.write(struct.pack('<H', len(b)))
            f.write(b)
            
    print(f"Exported {max_id + 1} tokens to {out_path}")

if __name__ == '__main__':
    main()
