# compact_lmdb.py
import os, lmdb, argparse

def compact(src_env_dir, dst_env_dir, headroom=1.10, readahead=False):
    os.makedirs(dst_env_dir, exist_ok=True)

    # Open source read-only
    src = lmdb.open(src_env_dir, readonly=True, lock=False,
                    readahead=readahead, max_readers=4096)

    # Make a compact copy (no holes)
    # (If your py-lmdb is ancient and doesn't support compact=True, remove it.)
    src.copy(dst_env_dir, compact=True)
    src.close()

    # Re-open destination and add a bit of growth headroom
    # (no need for max_dbs here)
    dst = lmdb.open(dst_env_dir, lock=False)
    info = dst.info()
    stat = dst.stat()
    used_bytes = (info["last_pgno"] + 1) * stat["psize"]
    dst.set_mapsize(int(used_bytes * headroom))
    dst.sync()
    dst.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="path/to/source_xxx.lmdb")
    ap.add_argument("dst", help="path/to/dest_xxx.lmdb (will be created)")
    ap.add_argument("--headroom", type=float, default=1.10)
    ap.add_argument("--readahead", action="store_true",
                    help="enable kernel readahead (defaults off for network filesystems)")
    args = ap.parse_args()
    compact(args.src, args.dst, args.headroom, args.readahead)
    print("Compacted LMDB at:", args.dst)