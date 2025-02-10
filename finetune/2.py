# 读取原始文本文件
with open('/mnt/petrelfs/yangdanni/code/cogvideo_sft/CogVideo/finetune/train_data/hug_debug/videos.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
 
# 去掉每行中的 ".mp4" 后缀
new_lines = []
for line in lines:
    # 假设每行都是一个可能包含 ".mp4" 后缀的字符串
    new_line = line.rstrip().rsplit('.', 1)[0] + '\n'  # 去掉后缀并添加换行符
    new_lines.append(new_line)
 
# 将修改后的内容写入新的文本文件
with open('/mnt/petrelfs/yangdanni/code/cogvideo_sft/CogVideo/finetune/train_data/hug_debug/lose_videos.txt', 'w', encoding='utf-8') as file:
    file.writelines(new_lines)
 
print("处理完成，新的文本文件已保存为 new_file.txt")