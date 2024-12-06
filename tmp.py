import base64
import gzip

# 从日志中获取的 data 字符串
data_base64 = 'H4sIAAAAAAAA//JIzcnJVwjPL8pJUQQEAAD//6McKRwMAAAA'
compressed_data = base64.b64decode(data_base64)
decompressed_data = gzip.decompress(compressed_data)
string_value = decompressed_data.decode('utf-8')
print(f"{string_value}")