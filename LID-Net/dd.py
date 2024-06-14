import os


def rename_images_in_folder(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    # 过滤出图片文件（假设图片文件有以下扩展名）
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
    images = [f for f in files if f.lower().endswith(image_extensions)]

    # 按顺序重命名图片
    for i, filename in enumerate(images):
        # 构造新的文件名
        new_filename = f"test_{i + 1}.png"
        # 获取文件的完整路径
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)

        # 重命名文件
        os.rename(old_filepath, new_filepath)
        print(f"Renamed '{old_filepath}' to '{new_filepath}'")


# 指定要重命名图片的文件夹路径
folder_path = './test'
rename_images_in_folder(folder_path)
