import argparse
from collections import defaultdict
import gc
import glob
import gzip
import json
import multiprocessing as mp
import numpy as np
import os
import resource
import shutil
import sys
import time
from tqdm import tqdm
import transformers
import struct
import zstandard as zstd
transformers.utils.logging.set_verbosity(40) # suppress warnings

# TODO: change to .dataset to import from local utility function (note: this added code if from the polypythia seed huggingface repo)
import struct
from functools import lru_cache
from itertools import accumulate
from pathlib import Path
from torch.utils.data import Dataset

dtypes = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float32, 7: np.float64, 8: np.uint16}


def index_file_path(prefix_path):
    return prefix_path + ".idx"


def data_file_path(prefix_path):
    return prefix_path + ".bin"


def _warmup_mmap_file(path):
    with open(path, "rb") as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class MMapIndexedDataset(Dataset):
    class Index:
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @classmethod
        def writer(cls, path, dtype):
            class _Writer:
                def __enter__(self):
                    self._file = open(path, "wb")  # noqa: SIM115

                    # Write Magic string so we can check the file format then opening it again.
                    self._file.write(cls._HDR_MAGIC)
                    # Write version number
                    # Little endian unsigned 64 Bit integer
                    self._file.write(struct.pack("<Q", 1))
                    # Little endian unsigned 8 Bit integer
                    self._file.write(struct.pack("<B", code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    pointers = np.zeros(len(sizes), dtype=np.int64)
                    sizes = np.array(sizes, dtype=np.int64)

                    np.cumsum(sizes[:-1], out=pointers[1:])
                    pointers = pointers * dtype().itemsize
                    return pointers

                def write(self, sizes, doc_idx):
                    pointers = self._get_pointers(sizes)

                    # Little endian unsigned 64 Bit integer
                    self._file.write(struct.pack("<Q", len(sizes)))
                    # Little endian unsigned 64 Bit integer
                    self._file.write(struct.pack("<Q", len(doc_idx)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order="C"))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order="C"))
                    del pointers

                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order="C"))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path, skip_warmup=False):
            with open(path, "rb") as stream:
                magic_test = stream.read(9)
                assert magic_test == self._HDR_MAGIC, (
                    "Index file doesn't match expected format. " "Make sure that --dataset-impl is configured properly."
                )
                # Little endian unsigned 64 Bit integer
                version = struct.unpack("<Q", stream.read(8))
                assert version == (1,)

                # Little endian unsigned 8 Bit integer
                (dtype_code,) = struct.unpack("<B", stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack("<Q", stream.read(8))[0]
                self._doc_count = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                print("    warming up index mmap file...")
                _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            print("    reading sizes...")
            self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32, count=self._len, offset=offset)
            print("    reading pointers...")
            self._pointers = np.frombuffer(
                self._bin_buffer, dtype=np.int64, count=self._len, offset=offset + self._sizes.nbytes
            )
            print("    reading document index...")
            self._doc_idx = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._doc_count,
                offset=offset + self._sizes.nbytes + self._pointers.nbytes,
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)  # noqa: B019
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup=False):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path, skip_warmup):
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)

        if not skip_warmup:
            print("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        print("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode="r", order="C")
        print("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr)
            sents = np.split(np_array, offsets[:-1])
            return sents

    def get(self, idx, offset=0, length=None):
        """Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr)
        return np_array

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))


tokenizer = None

def load_file(path):

    if path.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            lines = f.readlines()
    elif path.endswith('.zst'):
        with open(path, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                decompressed_data = reader.read().decode('utf-8')
            lines = decompressed_data.split('\n')
            if lines[-1] == '':
                lines = lines[:-1]
    elif path.endswith('.jsonl'):
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
    else:
        raise ValueError(f'Unknown file type: {path}')
    return lines

def parse_line(args, line, rel_path, linenum):

    global tokenizer
    meta = json.loads(line.strip('\n'))
    if tokenizer is None:
        token_ids = meta['text'].encode('utf-8')
        if args.reversed:
            token_ids = token_ids[::-1].copy()
        data = token_ids
    else:
        token_ids = tokenizer.encode(meta['text'])
        if args.reversed:
            token_ids = token_ids[::-1].copy()
        data = np.array(token_ids, dtype=args.token_dtype).view(np.uint8).tobytes()
    del meta['text']
    data = args.doc_sep + data
    meta = (json.dumps({'path': rel_path, 'linenum': linenum, 'metadata': meta}) + '\n').encode('utf-8')
    return data, meta, token_ids

def prepare_fewfiles(args):

    ds_path = os.path.join(args.save_dir, f'tokenized')
    od_path = os.path.join(args.save_dir, f'offset')
    mt_path = os.path.join(args.save_dir, f'metadata')
    om_path = os.path.join(args.save_dir, f'metaoff')
    ug_path = os.path.join(args.save_dir, f'unigram')
    if all([os.path.exists(path) for path in [ds_path, od_path]]):
        print('Step 1 (prepare): Skipped. All files already exist.', flush=True)
        return

    print('Step 1 (prepare): Starting ...', flush=True)
    start_time = time.time()

    data_paths = list(sorted(glob.glob(f'{args.data_dir}/**/*.json*', recursive=True)))

    ds_fout = open(ds_path, 'wb')
    od_fout = open(od_path, 'wb')
    if args.add_metadata:
        mt_fout = open(mt_path, 'wb')
        om_fout = open(om_path, 'wb')
    if args.add_unigram:
        ug_fout = open(ug_path, 'w')
        unigram_counts = defaultdict(int)

    with mp.get_context('fork').Pool(args.cpus) as p:
        od = 0
        if args.add_metadata:
            om = 0
        for data_path in tqdm(data_paths):
            rel_path = data_path[len(args.data_dir)+1:]
            lines = load_file(data_path)
            for offset in range(0, len(lines), args.batch_size):
                batch_lines = lines[offset:min(offset+args.batch_size, len(lines))]
                results = p.starmap(parse_line, [(args, line, rel_path, offset+i) for i, line in enumerate(batch_lines)])
                for (data, meta, token_ids) in results:
                    ds_fout.write(data)
                    od_fout.write(np.array([od], dtype=np.uint64).view(np.uint8).tobytes())
                    od += len(data)
                    if args.add_metadata:
                        mt_fout.write(meta)
                        om_fout.write(np.array([om], dtype=np.uint64).view(np.uint8).tobytes())
                        om += len(meta)
                    if args.add_unigram:
                        for token_id in token_ids:
                            unigram_counts[token_id] += 1
                        unigram_counts[256**args.token_width-1] += 1
                del results
            del lines
            gc.collect()
    gc.collect()

    ds_fout.close()
    od_fout.close()
    if args.add_metadata:
        mt_fout.close()
        om_fout.close()
    if args.add_unigram:
        for token_id, count in sorted(unigram_counts.items()):
            ug_fout.write(f'{token_id} {count}\n')
        ug_fout.close()

    end_time = time.time()
    print(f'Step 1 (prepare): Done. Took {end_time-start_time:.2f} seconds', flush=True)

def prepare_pretokenized(args):
    ds_path = os.path.join(args.save_dir, f'tokenized')
    od_path = os.path.join(args.save_dir, f'offset')
    mt_path = os.path.join(args.save_dir, f'metadata')
    om_path = os.path.join(args.save_dir, f'metaoff')
    ug_path = os.path.join(args.save_dir, f'unigram')
    if all([os.path.exists(path) for path in [ds_path, od_path]]):
        print('Step 1 (prepare): Skipped. All files already exist.', flush=True)
        return

    print('Step 1 (prepare): Starting ...', flush=True)
    start_time = time.time()

    bin_path = list(glob.glob(f'{args.data_dir}/*.bin', recursive=True))
    assert len(bin_path) == 1, f"Found {len(bin_path)} bin files in {args.data_dir}. Expected 1."
    bin_path = bin_path[0]
    assert os.path.exists(bin_path.replace(".bin", ".idx")), f"Index file {bin_path.replace('.bin', '.idx')} does not exist."

    ds = MMapIndexedDataset(bin_path.replace(".bin", ""), skip_warmup=True)
    

    ds_fout = open(ds_path, 'wb')
    od_fout = open(od_path, 'wb')
    if args.add_metadata:
        mt_fout = open(mt_path, 'wb')
        om_fout = open(om_path, 'wb')
    if args.add_unigram:
        ug_fout = open(ug_path, 'w')
        unigram_counts = defaultdict(int)

    assert args.token_width == np.dtype(ds._index.dtype).itemsize, f"Token width {args.token_width} does not match dataset token width {np.dtype(ds._index.dtype).itemsize}."

    od = 0
    if args.add_metadata:
        om = 0
    for i, data in enumerate(tqdm(ds)):
        if args.reversed:
            data = data[::-1]
        blob = data.astype(ds._index.dtype, copy=False).tobytes()
        blob = args.doc_sep + blob
        ds_fout.write(blob)
        od_fout.write(struct.pack('<Q', od))
        od += len(blob)
        if args.add_metadata:
            meta = (json.dumps({'doc_id': i}) + '\n').encode('utf-8')
            mt_fout.write(meta)
            om_fout.write(struct.pack('<Q', om))
            om += len(meta)
        if args.add_unigram:
            for token_id in data:
                unigram_counts[token_id] += 1
            unigram_counts[256**args.token_width-1] += 1
    gc.collect()
    ds_fout.close()
    od_fout.close()

    if args.add_unigram:
        for token_id, count in sorted(unigram_counts.items()):
            ug_fout.write(f'{token_id} {count}\n')
        ug_fout.close()

    end_time = time.time()
    print(f'Step 1 (prepare): Done. Took {end_time-start_time:.2f} seconds', flush=True)

def prepare_manyfiles_map(args, filenum, path):

    rel_path = path[len(args.data_dir)+1:]
    lines = load_file(path)

    ds_fout = open(f'{args.temp_dir}/files/tokenized.{filenum}', 'wb')
    od_fout = open(f'{args.temp_dir}/files/offset.{filenum}', 'wb')
    if args.add_metadata:
        mt_fout = open(f'{args.temp_dir}/files/metadata.{filenum}', 'wb')
        om_fout = open(f'{args.temp_dir}/files/metaoff.{filenum}', 'wb')
    if args.add_unigram:
        ug_fout = open(f'{args.temp_dir}/files/unigram.{filenum}', 'w')
        unigram_counts = defaultdict(int)
    od = 0
    if args.add_metadata:
        om = 0

    for linenum, line in enumerate(lines):
        data, meta, token_ids = parse_line(args, line, rel_path, linenum)
        ds_fout.write(data)
        od_fout.write(np.array([od], dtype=np.uint64).view(np.uint8).tobytes())
        od += len(data)
        if args.add_metadata:
            mt_fout.write(meta)
            om_fout.write(np.array([om], dtype=np.uint64).view(np.uint8).tobytes())
            om += len(meta)
        if args.add_unigram:
            for token_id in token_ids:
                unigram_counts[token_id] += 1
            unigram_counts[256**args.token_width-1] += 1

    ds_fout.close()
    od_fout.close()
    if args.add_metadata:
        mt_fout.close()
        om_fout.close()
    if args.add_unigram:
        for token_id, count in sorted(unigram_counts.items()):
            ug_fout.write(f'{token_id} {count}\n')
        ug_fout.close()

def prepare_manyfiles_reduce(args, filenum, data_prev_bytes, meta_prev_bytes, offset_prev_bytes):

    ds_fout = open(f'{args.save_dir}/tokenized', 'rb+') # do not truncate
    ds_fout.seek(data_prev_bytes)
    od_fout = open(f'{args.save_dir}/offset', 'rb+')
    od_fout.seek(offset_prev_bytes)
    if args.add_metadata:
        mt_fout = open(f'{args.save_dir}/metadata', 'rb+')
        mt_fout.seek(meta_prev_bytes)
        om_fout = open(f'{args.save_dir}/metaoff', 'rb+')
        om_fout.seek(offset_prev_bytes)

    with open(f'{args.temp_dir}/files/tokenized.{filenum}', 'rb') as f:
        ds_fout.write(f.read())
    with open(f'{args.temp_dir}/files/offset.{filenum}', 'rb') as f:
        offsets = np.frombuffer(f.read(), dtype=np.uint8).view(np.uint64).copy()
        offsets += data_prev_bytes
        od_fout.write(offsets.view(np.uint8).tobytes())

    if args.add_metadata:
        with open(f'{args.temp_dir}/files/metadata.{filenum}', 'rb') as f:
            mt_fout.write(f.read())
        with open(f'{args.temp_dir}/files/metaoff.{filenum}', 'rb') as f:
            offsets = np.frombuffer(f.read(), dtype=np.uint8).view(np.uint64).copy()
            offsets += meta_prev_bytes
            om_fout.write(offsets.view(np.uint8).tobytes())

    ds_fout.close()
    od_fout.close()
    if args.add_metadata:
        mt_fout.close()
        om_fout.close()

def prepare_manyfiles(args):

    ds_path = os.path.join(args.save_dir, f'tokenized')
    od_path = os.path.join(args.save_dir, f'offset')
    mt_path = os.path.join(args.save_dir, f'metadata')
    om_path = os.path.join(args.save_dir, f'metaoff')
    ug_path = os.path.join(args.save_dir, f'unigram')
    if all([os.path.exists(path) for path in [ds_path, od_path]]):
        print('Step 1 (prepare): Skipped. All files already exist.', flush=True)
        return

    print('Step 1 (prepare): Starting ...', flush=True)
    start_time = time.time()

    data_paths = list(sorted(glob.glob(f'{args.data_dir}/**/*.json*', recursive=True), key=lambda x: x.replace("crawl=", "")))

    with mp.get_context('fork').Pool(args.cpus) as p:
        shutil.rmtree(f'{args.temp_dir}/files', ignore_errors=True)
        os.makedirs(f'{args.temp_dir}/files')
        _ = p.starmap(prepare_manyfiles_map, [(args, filenum, data_path) for (filenum, data_path) in enumerate(data_paths)])

        tasks = []
        data_prev_bytes = 0
        meta_prev_bytes = 0
        offset_prev_bytes = 0
        for filenum in range(len(data_paths)):
            tasks.append((args, filenum, data_prev_bytes, meta_prev_bytes, offset_prev_bytes))
            data_prev_bytes += os.path.getsize(f'{args.temp_dir}/files/tokenized.{filenum}')
            if args.add_metadata:
                meta_prev_bytes += os.path.getsize(f'{args.temp_dir}/files/metadata.{filenum}')
            offset_prev_bytes += os.path.getsize(f'{args.temp_dir}/files/offset.{filenum}')

        with open(ds_path, 'wb') as f:
            data_file_size = data_prev_bytes
            f.truncate(data_file_size)
        with open(od_path, 'wb') as f:
            data_offset_file_size = offset_prev_bytes
            f.truncate(data_offset_file_size)
        if args.add_metadata:
            with open(mt_path, 'wb') as f:
                meta_file_size = meta_prev_bytes
                f.truncate(meta_file_size)
            with open(om_path, 'wb') as f:
                meta_offset_file_size = offset_prev_bytes
                f.truncate(meta_offset_file_size)

        _ = p.starmap(prepare_manyfiles_reduce, tasks)

        if args.add_unigram:
            unigram_counts = defaultdict(int)
            for filenum in range(len(data_paths)):
                with open(f'{args.temp_dir}/files/unigram.{filenum}', 'r') as f:
                    for line in f:
                        token_id, count = line.split()
                        unigram_counts[int(token_id)] += int(count)
            with open(ug_path, 'w') as f:
                for token_id, count in sorted(unigram_counts.items()):
                    f.write(f'{token_id} {count}\n')

        shutil.rmtree(f'{args.temp_dir}/files')

    end_time = time.time()
    print(f'Step 1 (prepare): Done. Took {end_time-start_time:.2f} seconds', flush=True)

def prepare(args):

    global tokenizer
    if args.tokenizer is None:
        tokenizer = None
    elif args.tokenizer == 'gpt2':
        tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2', use_fast=False, add_bos_token=False, add_eos_token=False)
    elif args.tokenizer == 'llama':
        tokenizer = transformers.AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token=os.environ.get('HF_TOKEN'), use_fast=False, add_bos_token=False, add_eos_token=False) # The fast tokenizer seems unbearably slow ...
    elif args.tokenizer == 'olmo':
        tokenizer = transformers.AutoTokenizer.from_pretrained("allenai/OLMo-7B", add_bos_token=False, add_eos_token=False)
        # # The following is a faster version, but the result is a bit different
        # from dolma.tokenizer import Tokenizer
        # tokenizer = Tokenizer.from_pretrained('allenai/gpt-neox-olmo-dolma-v1_5', bos_token_id=None, eos_token_id=None, pad_token_id=1, segment_before_tokenization=True)
    elif args.tokenizer == 'gpt-neox-20b':
        tokenizer = transformers.AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b', trust_remote_code=True, use_fast=True, add_bos_token=False, add_eos_token=False)
    else:
        raise ValueError(f'Unknown tokenizer: {args.tokenizer}')

    if args.is_pretokenized:
        prepare_pretokenized(args)
    data_paths = list(sorted(glob.glob(f'{args.data_dir}/**/*.json*', recursive=True)))
    if len(data_paths) < args.cpus:
        prepare_fewfiles(args)
    else:
        prepare_manyfiles(args)

def build_sa(args):

    ds_path = os.path.join(args.save_dir, f'tokenized')
    sa_path = ds_path.replace('tokenized', 'table')
    if os.path.exists(sa_path):
        print(f'Step 2 (build_sa): Skipped. File already exists.', flush=True)
        return

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    print('Step 2 (build_sa): Starting ...', flush=True)
    start_time_all = time.time()

    # -------- Step 2.1 (make-part) -------- #

    print(f'Step 2.1 (make-part): Starting ...', flush=True)
    start_time = time.time()

    ds_size = os.path.getsize(ds_path)
    ratio = int(np.ceil(np.log2(ds_size) / 8))
    mem_bytes = args.mem * 1024**3
    num_job_batches = 1
    while num_job_batches * (mem_bytes // (12 if args.token_width == 1 else 8)) < ds_size:
        num_job_batches *= 2
    parallel_jobs = args.cpus
    total_jobs = num_job_batches * parallel_jobs
    print(f'Using {num_job_batches} batches of {parallel_jobs} jobs each, for a total of {total_jobs} jobs.', flush=True)

    S = ds_size // total_jobs
    # Make sure that parts contain whole tokens
    if S % args.token_width != 0:
        S += args.token_width - S % args.token_width

    parts_dir = os.path.join(args.temp_dir, f'parts')
    shutil.rmtree(parts_dir, ignore_errors=True)
    os.makedirs(parts_dir)

    for batch_start in tqdm(list(range(0, total_jobs, parallel_jobs))):
        batch_end = min(batch_start+parallel_jobs, total_jobs)
        batch_ranges = []
        for i in range(batch_start, batch_end):
            s, e = i*S, min((i+1)*S+args.hack, ds_size)
            batch_ranges.append((s, e))
        pipes = []
        for (s, e) in batch_ranges:
            pipes.append(os.popen(f'./rust_indexing make-part --data-file {ds_path} --parts-dir {parts_dir} --start-byte {s} --end-byte {e} --ratio {ratio} --token-width {args.token_width}'))
        [pipe.read() for pipe in pipes]
        if any([pipe.close() is not None for pipe in pipes]):
            print('Step 2.1 (make-part): Something went wrong', flush=True)
            exit(1)

    end_time = time.time()
    print(f'Step 2.1 (make-part): Done. Took {end_time-start_time:.2f} seconds', flush=True)

    # -------- Step 2.2 (merge) -------- #

    print(f'Step 2.2 (merge): Starting ...', flush=True)
    start_time = time.time()

    merged_dir = os.path.join(args.temp_dir, f'merged')
    shutil.rmtree(merged_dir, ignore_errors=True)
    os.makedirs(merged_dir)

    pipe = os.popen(f'./rust_indexing merge --data-file {ds_path} --parts-dir {parts_dir} --merged-dir {merged_dir} --num-threads {args.cpus} --hacksize {args.hack} --ratio {ratio} --token-width {args.token_width}')
    pipe.read()
    if pipe.close() is not None:
        print('Step 2.2 (merge): Something went wrong', flush=True)
        exit(1)

    shutil.rmtree(parts_dir)

    end_time = time.time()
    print(f'Step 2.2 (merge): Done. Took {end_time-start_time:.2f} seconds', flush=True)

    # -------- Step 2.3 (concat) -------- #

    print(f'Step 2.3 (concat): Starting ...', flush=True)
    start_time = time.time()

    pipe = os.popen(f'./rust_indexing concat --data-file {ds_path} --merged-dir {merged_dir} --merged-file {sa_path} --num-threads {args.cpus} --ratio {ratio} --token-width {args.token_width}')
    pipe.read()
    if pipe.close() is not None:
        print('Step 2.3 (concat): Something went wrong', flush=True)
        exit(1)

    shutil.rmtree(merged_dir)

    end_time = time.time()
    print(f'Step 2.3 (concat): Done. Took {end_time-start_time:.2f} seconds', flush=True)

    end_time_all = time.time()
    print(f'Step 2 (build_sa): Done. Took {end_time_all-start_time_all:.2f} seconds', flush=True)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the raw text corpus. Must be absolute path.')
    parser.add_argument('--temp_dir', type=str, default=None, help='Directory where temporary indexing files are stored. Must be absolute path.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory where the final index files are stored. Must be absolute path.')
    parser.add_argument('--version', type=int, default=6, choices=[6], help='Version of the index.')
    parser.add_argument('--reversed', default=False, action='store_true', help='Whether to reverse the tokens in each document.')
    parser.add_argument('--tokenizer', type=str, default=None, choices=[None, 'gpt2', 'llama', 'olmo', 'gpt-neox-20b'])
    parser.add_argument('--token_dtype', type=str, default='u16', choices=['u8', 'u16', 'u32'], help='Data type for tokens.')
    parser.add_argument('--add_metadata', default=False, action='store_true', help='Whether to store document metadata in the index.')
    parser.add_argument('--add_unigram', default=False, action='store_true', help='Whether to precompute unigram counts.')
    parser.add_argument('--batch_size', type=int, default=65536, help='Batch size for tokenization.')
    parser.add_argument('--hack', type=int, default=100000, help='Hack size in bytes.')
    parser.add_argument('--cpus', type=int, default=mp.cpu_count(), help='Number of CPU cores available to the program.')
    parser.add_argument('--mem', type=int, required=True, help='Amount of memory in GiB available to the program.')
    parser.add_argument('--ulimit', type=int, default=1048576, help='Maximum number of open files allowed.')
    parser.add_argument('--is_pretokenized', default=False, action='store_true', help='Whether the input files are already pretokenized.')
    args = parser.parse_args()

    if args.temp_dir is None:
        args.temp_dir = args.save_dir
    args.data_dir = args.data_dir.rstrip('/')
    args.temp_dir = args.temp_dir.rstrip('/')
    args.save_dir = args.save_dir.rstrip('/')

    assert args.batch_size > 0
    assert args.cpus > 0

    if args.token_dtype == 'u8':
        args.token_width = 1
        args.doc_sep = b'\xff'
    elif args.token_dtype == 'u16':
        args.token_width = 2
        args.doc_sep = b'\xff\xff'
    elif args.token_dtype == 'u32':
        args.token_width = 4
        args.doc_sep = b'\xff\xff\xff\xff'
    else:
        raise ValueError(f'Unknown token_dtype: {args.token_dtype}')

    assert os.path.exists(args.data_dir)
    os.makedirs(args.temp_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    assert sys.byteorder == 'little'
    resource.setrlimit(resource.RLIMIT_NOFILE, (args.ulimit, args.ulimit))

    prepare(args)
    build_sa(args)

if __name__ == '__main__':
    main()
