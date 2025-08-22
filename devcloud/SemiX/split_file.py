import os
import argparse

def split_file(file_path, chunk_size_mb):
    """Splits a file into smaller chunks with correct naming."""
    try:
        file_size = os.path.getsize(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    num_chunks = int(file_size / chunk_size_bytes) + (1 if file_size % chunk_size_bytes != 0 else 0)

    # Correct filename logic to ensure original filename is preserved
    base_name = os.path.basename(file_path)

    with open(file_path, 'rb') as f_in:
        for i in range(num_chunks):
            chunk_name = f"{base_name}.part{i + 1}"
            chunk_path = os.path.join(os.path.dirname(file_path), chunk_name)
            
            with open(chunk_path, 'wb') as f_out:
                bytes_to_write = min(chunk_size_bytes, file_size - (i * chunk_size_bytes))
                f_out.write(f_in.read(bytes_to_write))
            print(f"Created chunk: {chunk_path}")

def merge_file(file_path):
    """Merges split file chunks back into a single file."""
    base_name = os.path.basename(file_path)
    parent_dir = os.path.dirname(file_path) or '.'
    
    chunk_files = []
    i = 1
    while True:
        chunk_name = f"{base_name}.part{i}"
        chunk_path = os.path.join(parent_dir, chunk_name)
        if os.path.exists(chunk_path):
            chunk_files.append(chunk_path)
            i += 1
        else:
            break
            
    if not chunk_files:
        print("Error: No file chunks found to merge.")
        return

    with open(file_path, 'wb') as f_out:
        for chunk_name in chunk_files:
            with open(chunk_name, 'rb') as f_in:
                f_out.write(f_in.read())
            print(f"Merged chunk: {chunk_name}")
    print(f"\nFile successfully merged into {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split or merge a file.")
    subparsers = parser.add_subparsers(dest="action", required=True, help="Action to perform")

    # Split sub-command
    split_parser = subparsers.add_parser("split", help="Split a file into smaller chunks.")
    split_parser.add_argument("file_path", help="Path to the file to split.")
    split_parser.add_argument("chunk_size_mb", type=int, help="Size of each chunk in megabytes (MB).")

    # Merge sub-command
    merge_parser = subparsers.add_parser("merge", help="Merge file chunks back into a single file.")
    merge_parser.add_argument("file_path", help="Path to the original file to merge (e.g., your_model.bin).")

    args = parser.parse_args()

    if args.action == "split":
        split_file(args.file_path, args.chunk_size_mb)
    elif args.action == "merge":
        merge_file(args.file_path)
