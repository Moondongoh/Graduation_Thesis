import os

def create_directory_tree_txt(start_path='.', output_file='directory_tree.txt'):
    """
    현재 폴더 구조를 트리 형식으로 TXT 파일에 작성하는 함수입니다.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"현재 폴더 구조 ({start_path}):\n\n")

        for root, dirs, files in os.walk(start_path):
            # 현재 경로의 깊이를 계산하여 들여쓰기 수준을 정합니다.
            level = root.replace(start_path, '').count(os.sep)
            indent = ' ' * 4 * (level)

            # 현재 디렉토리 이름과 들여쓰기를 파일에 작성합니다.
            f.write(f"{indent}{os.path.basename(root)}/\n")

            subindent = ' ' * 4 * (level + 1)
            # 현재 디렉토리의 파일들을 작성합니다.
            for file in files:
                f.write(f"{subindent}{file}\n")

    return f"성공적으로 '{output_file}' 파일을 생성했습니다."

# 현재 폴더 구조를 'directory_tree.txt' 파일로 작성합니다.
create_directory_tree_txt()