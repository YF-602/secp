import os
import re
def count_code_lines(lines):
   count = 0
   for line in lines:
       if line.strip() != '' and not re.match(r'^\s*#', line): # 排除空行和注释
           count += 1
   return count
def calculate_code_lines(file_path):
   with open(file_path, 'r') as file:
       lines = file.readlines()
   return count_code_lines(lines)
def calculate_project_code_lines(folder_path):
   total_lines = 0
   for root, dirs, files in os.walk(folder_path):
       for file in files:
           if file.endswith('.py'): # 仅统计 Python 文件
               file_path = os.path.join(root, file)
               total_lines += calculate_code_lines(file_path)
   return total_lines
# 示例用法
if __name__ == '__main__':
   project_folder = './utils' # 替换为你的项目路径
   total_lines = calculate_project_code_lines(project_folder)
   print("总代码行数:", total_lines)