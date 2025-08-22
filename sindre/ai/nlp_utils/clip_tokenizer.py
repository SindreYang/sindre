import gzip
import html
import os
from functools import lru_cache
import ftfy  # 用于修复文本编码和格式问题
import regex as re  # 增强版正则，支持Unicode匹配
import torch.nn as nn  # 导入nn模块，便于后续作为模型组件使用
from typing import Union, List
import torch
import urllib
from io import BytesIO

@lru_cache()  # 缓存结果，避免重复计算（装饰器作用于无状态函数）
def default_bpe():
    """
    获取默认BPE词表文件路径（BPE：Byte Pair Encoding，字节对编码）
    词表文件与当前脚本在同一目录下，默认名为 "bpe_simple_vocab_16e6.txt.gz"

    Returns:
        str: BPE词表文件的绝对路径
    """
    # 获取当前脚本的绝对路径 → 取其所在目录 → 拼接词表文件名
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()  # 缓存映射表，避免重复构建
def bytes_to_unicode():
    """
    构建UTF-8字节与Unicode字符的双向映射表
    核心目的：
    1. 将不可见/控制字符映射为可见Unicode字符，避免BPE处理时出错
    2. 覆盖所有256个UTF-8字节，确保无UNK（未识别字符）

    Returns:
        dict: key=UTF-8字节值（0-255），value=对应的Unicode字符
    """
    # 1. 先加入可打印ASCII字符（!到~）和部分拉丁字符（¡到¬、®到ÿ）
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs.copy()  # cs用于存储对应的Unicode编码值

    # 2. 补充未覆盖的字节（0-255中不在bs里的部分），映射到高Unicode区间（256+）
    n = 0
    for b in range(2**8):  # 遍历所有256个UTF-8字节
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)  # 高区间避免与原有字符冲突
            n += 1

    # 3. 将Unicode编码值转为字符，构建映射字典
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    提取单词中所有连续的字符对（用于BPE合并规则）
    单词以「字符元组」形式输入（字符可为单字节或已合并的多字节）

    Args:
        word (tuple): 单词的字符元组，例：("h", "e", "l", "l", "o")

    Returns:
        set: 字符对集合，例：{("h","e"), ("e","l"), ("l","l"), ("l","o")}
    """
    pairs = set()
    prev_char = word[0]  # 记录前一个字符
    for char in word[1:]:  # 遍历后续字符，生成连续对
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    """
    基础文本清洗：修复编码错误、解码HTML实体

    Args:
        text (str): 原始输入文本

    Returns:
        str: 清洗后的文本（去除首尾空格）
    """
    text = ftfy.fix_text(text)  # 修复文本编码（如UTF-8乱码、全角半角转换）
    text = html.unescape(html.unescape(text))  # 解码HTML实体（如&amp;→&，&lt;→<）
    return text.strip()  # 去除首尾空格


def whitespace_clean(text):
    """
    空白符清洗：将连续空白符（空格、换行、制表符等）替换为单个空格

    Args:
        text (str): 输入文本

    Returns:
        str: 清洗后的文本（去除首尾空格）
    """
    text = re.sub(r"\s+", " ", text)  # 正则匹配所有连续空白符，替换为单个空格
    text = text.strip()
    return text


class SimpleTokenizer(nn.Module):
    """
    基于BPE（Byte Pair Encoding）的文本分词器
    参考CLIP的SimpleTokenizer实现，支持文本→token编码、token→文本解码
    核心功能：处理多语言Unicode文本，无UNK字符，适配预训练模型输入格式
    """
    def __init__(self, bpe_path: str = default_bpe()):
        """
        初始化分词器，加载BPE词表并构建编码/解码所需组件

        Args:
            bpe_path (str): BPE合并规则词表的路径，默认使用同目录下的16M词表
        """
        super().__init__()  # 继承nn.Module，可作为模型组件使用

        # 1. 初始化字节-Unicode映射（编码/解码基础）
        self.byte_encoder = bytes_to_unicode()  # 字节→Unicode字符
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}  # 反向映射：Unicode→字节

        # 2. 加载BPE合并规则（从压缩词表文件）
        # 读取gzip压缩文件 → 解码为UTF-8 → 按行分割为合并规则
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        # 截取有效规则：跳过首行注释，取前49152-256-2+1条（CLIP词表标准长度）
        merges = merges[1 : 49152 - 256 - 2 + 1]
        # 将每条规则从字符串转为元组（例："h e" → ("h", "e")）
        self.merges = [tuple(merge.split()) for merge in merges]

        # 3. 构建BPE词汇表（vocab）和编码/解码字典
        # 初始词汇：所有字节对应的Unicode字符 + 带结束符"</w>"的字符（标记单词结尾）
        base_vocab = list(bytes_to_unicode().values())
        vocab = base_vocab + [v + "</w>" for v in base_vocab]
        # 加入所有BPE合并后的字符对（扩展词汇表）
        for merge in self.merges:
            vocab.append("".join(merge))
        # 加入特殊标记：文本开始/结束符（适配生成任务或批量处理）
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])

        # 构建编码字典（字符→索引）和解码字典（索引→字符）
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}

        # 4. BPE合并规则的优先级（rank越小，合并优先级越高）
        self.bpe_ranks = dict(zip(self.merges, range(len(self.merges))))

        # 5. 缓存常用特殊标记的BPE结果，避免重复计算
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>"
        }

        # 6. 文本分割正则：匹配特殊标记、常见缩写、字母、数字、符号
        # 覆盖场景：
        # - <|startoftext|>/<|endoftext|>：特殊标记
        # - 's/'t/'re等：英语常见缩写
        # - [\p{L}]：所有Unicode字母（支持多语言）
        # - [\p{N}]：所有Unicode数字
        # - [^\s\p{L}\p{N}]+：非空白、非字母、非数字的符号（如标点、表情）
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE  # 忽略大小写（如'S'和's'视为同一类）
        )

    def bpe(self, token):
        """
        对单个分词单元（token）应用BPE合并规则，生成最终BPE子词

        Args:
            token (str): 基础分词单元（如"hello"，已通过byte_encoder映射为Unicode）

        Returns:
            str: BPE合并后的子词字符串（用空格分隔子词，例："hel lo</w>"）
        """
        # 1. 先查缓存，避免重复计算
        if token in self.cache:
            return self.cache[token]

        # 2. 初始化单词：末尾添加"</w>"标记单词结束，转为元组便于处理
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        # 3. 提取初始字符对（用于后续合并）
        pairs = get_pairs(word)

        # 若没有字符对（单字符单词），直接返回带结束符的结果
        if not pairs:
            return token + "</w>"

        # 4. 迭代合并优先级最高的字符对，直到无法合并或只剩一个字符
        while True:
            # 找到当前优先级最高的字符对（rank最小）
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 若该字符对不在合并规则中，停止合并
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram  # 拆分当前要合并的字符对
            new_word = []  # 存储合并后的新字符序列
            i = 0  # 遍历指针

            # 遍历当前单词的字符，合并目标字符对
            while i < len(word):
                try:
                    # 找到当前指针后第一个"first"字符的位置
                    j = word.index(first, i)
                    new_word.extend(word[i:j])  # 将i到j之间的字符加入新序列
                    i = j  # 移动指针到j位置
                except ValueError:
                    # 未找到"first"，将剩余字符加入新序列，退出循环
                    new_word.extend(word[i:])
                    break

                # 若当前字符是"first"且下一个字符是"second"，合并为一个字符
                if (word[i] == first and
                        i < len(word) - 1 and
                        word[i + 1] == second):
                    new_word.append(first + second)
                    i += 2  # 跳过已合并的两个字符
                else:
                    new_word.append(word[i])
                    i += 1  # 移动指针到下一个字符

            # 更新单词为合并后的新序列，重新提取字符对
            word = tuple(new_word)
            if len(word) == 1:
                break  # 只剩一个字符，停止合并
            else:
                pairs = get_pairs(word)  # 重新计算字符对

        # 5. 将字符元组转为空格分隔的字符串，存入缓存并返回
        word_str = " ".join(word)
        self.cache[token] = word_str
        return word_str

    def encode(self, text):
        """
        将文本编码为token索引列表（模型输入格式）

        Args:
            text (str): 原始输入文本（支持多语言、特殊符号）

        Returns:
            list[int]: token索引列表，每个元素对应一个BPE子词的索引
        """
        # 1. 文本预处理：基础清洗→空白符清洗→转为小写
        text = whitespace_clean(basic_clean(text)).lower()

        # 2. 按正则分割文本为基础token（如单词、缩写、符号）
        bpe_tokens = []
        for raw_token in re.findall(self.pat, text):
            # 3. 将基础token转为UTF-8字节→再映射为Unicode字符（避免UNK）
            encoded_token = "".join(self.byte_encoder[b] for b in raw_token.encode("utf-8"))
            # 4. 应用BPE合并规则，拆分并转为token索引
            for bpe_subtoken in self.bpe(encoded_token).split(" "):
                bpe_tokens.append(self.encoder[bpe_subtoken])

        return bpe_tokens

    def decode(self, tokens):
        """
        将token索引列表解码为原始文本

        Args:
            tokens (list[int]): token索引列表（模型输出或编码结果）

        Returns:
            str: 解码后的文本（修复结束符，去除多余空格）
        """
        # 1. token索引→BPE子词字符（通过decoder）
        text = "".join([self.decoder[token] for token in tokens])
        # 2. Unicode字符→UTF-8字节→解码为文本（处理特殊字符）
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace")
        # 3. 替换单词结束符"</w>"为空格，返回最终文本
        return text.replace("</w>", " ")

    def forward(self, text_list):
        """
        扩展nn.Module的forward方法，支持批量文本编码（模型组件化时使用）

        Args:
            text_list (list[str]): 批量文本列表

        Returns:
            list[list[int]]: 批量token索引列表（每个元素对应一条文本的编码结果）
        """
        return [self.encode(text) for text in text_list]






class Tokenizer(SimpleTokenizer):
    """
    CLIP风格的文本Tokenizer，继承自基础BPE分词器SimpleTokenizer
    核心增强：自动添加文本开始/结束标记（SOT/EOT），支持批量tokenize并对齐到固定上下文长度
    适配CLIP系列模型的输入格式（默认上下文长度77）
    """
    def __init__(self, vocab_path: str):
        """
        初始化Tokenizer，复用SimpleTokenizer的BPE词表加载逻辑

        Args:
            vocab_path (str): BPE词表文件路径（或文件缓冲区对象）
                             与SimpleTokenizer的bpe_path参数功能一致，此处参数名统一为vocab_path
        """
        # 调用父类SimpleTokenizer的初始化方法，传入BPE词表路径
        SimpleTokenizer.__init__(self, bpe_path=vocab_path)

    def tokenize(
            self, texts: Union[str, List[str]], context_length: int = 77
    ):
        """
        将输入文本（单个/批量）转换为模型可接受的token索引张量
        自动处理：添加SOT/EOT标记、截断超长文本、用0填充到固定上下文长度

        Parameters
        ----------
        texts : Union[str, List[str]]
            待tokenize的输入，支持单个字符串或字符串列表（批量处理）
        context_length : int
            固定上下文长度，所有CLIP模型默认使用77（超出部分截断，不足部分补0）

        Returns
        -------
        torch.LongTensor
            二维token索引张量，形状为 [输入文本数量, context_length]
            每个元素对应一个BPE子词的索引，0表示填充位
        """
        # 1. 统一输入格式：若为单个字符串，转为长度为1的列表（便于批量处理）
        if isinstance(texts, str):
            texts = [texts]

        # 2. 获取特殊标记的token索引
        sot_token = self.encoder["<|startoftext|>"]  # SOT：文本开始标记（Start Of Text）
        eot_token = self.encoder["<|endoftext|>"]  # EOT：文本结束标记（End Of Text）

        # 3. 对每条文本进行编码：SOT标记 + 文本BPE编码 + EOT标记
        all_tokens = [
            [sot_token] + self.encode(text) + [eot_token]  # 调用父类SimpleTokenizer的encode方法
            for text in texts
        ]

        # 4. 初始化结果张量：用0填充（0为默认填充值），形状 [文本数量, context_length]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        # 5. 处理每条文本的token序列（截断/填充）
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                # 超长文本：截断到context_length，并确保最后一位是EOT标记（覆盖截断掉的原EOT）
                tokens = tokens[:context_length]
                tokens[-1] = eot_token  # 强制末尾为EOT，保证格式正确性
            # 将token序列写入结果张量（前len(tokens)位填实际索引，剩余位保持0填充）
            result[i, : len(tokens)] = torch.tensor(tokens)

        return result


def get_tokenizer(bpe_path_or_url: str="https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz") -> Tokenizer | None:
    """
    工厂函数：根据BPE词表的本地路径或网络URL，创建Tokenizer实例
    支持两种加载方式：本地文件读取、网络URL下载

    Args:
        bpe_path_or_url (str): BPE词表的本地路径（如"./bpe_vocab.txt.gz"）或网络URL（如"https://xxx/bpe_vocab.txt.gz"）

    Returns:
        Tokenizer | None: 成功创建的Tokenizer实例；若失败（如下载错误）则抛出异常，不返回None

    Raises:
        FileNotFoundError: 当从URL下载失败或本地文件无法读取时抛出
    """
    # 1. 解析输入字符串：判断是URL还是本地路径（通过URL协议头，如http/https）
    parsed_url = urllib.parse.urlparse(bpe_path_or_url)
    if parsed_url.scheme:  # 有协议头（如http、https），判定为URL
        try:
            # 2. 从URL下载词表文件：打开URL连接→读取二进制内容→存入BytesIO缓冲区（模拟文件对象）
            with urllib.request.urlopen(bpe_path_or_url) as response:
                file_buf = BytesIO(response.read())  # BytesIO支持文件式读写，适配Tokenizer的vocab_path参数
            # 3. 用缓冲区创建Tokenizer实例（Tokenizer可接受文件对象作为vocab_path）
            return Tokenizer(vocab_path=file_buf)
        except Exception as e:
            # 下载失败（如网络错误、URL无效），抛出明确的错误信息
            raise FileNotFoundError(
                f"从URL {bpe_path_or_url} 下载BPE词表失败，错误信息：{e}"
            ) from e  # from e 保留原始错误栈，便于调试
    else:  # 无协议头，判定为本地文件路径
        try:
            # 2. 读取本地词表文件：以二进制模式打开→存入BytesIO缓冲区
            with open(bpe_path_or_url, "rb") as f:
                file_buf = BytesIO(f.read())
            # 3. 创建Tokenizer实例
            return Tokenizer(vocab_path=file_buf)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"本地BPE词表文件 {bpe_path_or_url} 不存在"
            ) from None
        except Exception as e:
            raise FileNotFoundError(
                f"读取本地BPE词表文件 {bpe_path_or_url} 失败，错误信息：{e}"
            ) from e