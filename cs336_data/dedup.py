import os
import hashlib


def exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    """
    执行精确行去重，去除在整个语料库中出现超过一次的行。

    """
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)
    
    # 第一遍扫描：统计每行的出现次数
    line_counts = {}
    
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 使用行的哈希值作为键来节省内存
                line_hash = hashlib.md5(line.encode('utf-8')).hexdigest()
                line_counts[line_hash] = line_counts.get(line_hash, 0) + 1
    
    # 第二遍扫描：重写每个文件，只保留唯一的行
    for file_path in input_files:
        # 构建输出文件路径
        output_file_path = os.path.join(output_directory, os.path.basename(file_path))
        
        with open(file_path, 'r', encoding='utf-8') as input_file, \
             open(output_file_path, 'w', encoding='utf-8') as output_file:
            
            for line in input_file:
                # 检查这行是否只出现一次
                line_hash = hashlib.md5(line.encode('utf-8')).hexdigest()
                if line_counts[line_hash] == 1:
                    output_file.write(line)


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    raise NotImplementedError