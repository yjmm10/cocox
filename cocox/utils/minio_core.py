import os
from minio import Minio
from minio.error import S3Error

# 从环境变量获取MinIO配置
ak = os.environ.get("MINIO_ACCESS_KEY", "")
sk = os.environ.get("MINIO_SECRET_KEY", "")
endpoint = os.environ.get("MINIO_ENDPOINT", "")

# 初始化MinIO客户端
client = Minio(
    endpoint,
    access_key=ak,
    secret_key=sk,
    secure=False  # 如果使用HTTPS，设置为True
)

def upload_file(file_path, bucket_name, object_name):
    """
    上传单个文件到MinIO
    
    Args:
        file_path: 本地文件路径
        bucket_name: 存储桶名称
        object_name: 对象名称（MinIO中的路径）
    """
    try:
        # 确保bucket存在
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"创建存储桶 {bucket_name} 成功")
        
        # 上传文件
        client.fput_object(
            bucket_name, object_name, file_path,
        )
        print(f"文件 {file_path} 上传成功，存储为 {object_name}")
    except S3Error as err:
        print(f"上传失败: {err}")

def upload_directory(directory_path, bucket_name, object_prefix):
    """
    上传整个目录到MinIO
    
    Args:
        directory_path: 本地目录路径
        bucket_name: 存储桶名称
        object_prefix: 对象前缀（MinIO中的目录）
    """
    try:
        # 确保bucket存在
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"创建存储桶 {bucket_name} 成功")
        
        # 遍历目录中的所有文件
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                # 获取文件的完整路径
                file_path = os.path.join(root, file)
                # 计算相对路径
                rel_path = os.path.relpath(file_path, directory_path)
                # 构建对象名称
                obj_name = os.path.join(object_prefix, rel_path).replace("\\", "/")
                
                # 上传文件
                client.fput_object(
                    bucket_name, obj_name, file_path,
                )
                print(f"文件 {file_path} 上传成功，存储为 {obj_name}")
        
        print(f"目录 {directory_path} 上传完成")
    except S3Error as err:
        print(f"上传目录失败: {err}")

def download_file(bucket_name, object_name, download_path):
    """
    从MinIO下载单个文件
    
    Args:
        bucket_name: 存储桶名称
        object_name: 对象名称（MinIO中的路径）
        download_path: 下载目录路径
    """
    try:
        # 确保下载目录存在
        os.makedirs(download_path, exist_ok=True)
        
        # 构建本地文件路径
        local_file_path = os.path.join(download_path, os.path.basename(object_name))
        
        # 下载文件
        client.fget_object(bucket_name, object_name, local_file_path)
        print(f"文件 {object_name} 下载成功，保存为 {local_file_path}")
    except S3Error as err:
        print(f"下载失败: {err}")

def download_directory(bucket_name, prefix, download_path):
    """
    从MinIO下载目录
    
    Args:
        bucket_name: 存储桶名称
        prefix: 对象前缀（MinIO中的目录）
        download_path: 下载目录路径
    """
    try:
        # 确保下载目录存在
        os.makedirs(download_path, exist_ok=True)
        
        # 列出指定前缀的所有对象
        objects = client.list_objects(bucket_name, prefix=prefix, recursive=True)
        
        for obj in objects:
            # 构建本地文件路径
            rel_path = obj.object_name[len(prefix):].lstrip('/')
            local_file_path = os.path.join(download_path, rel_path)
            
            # 确保本地目录存在
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # 下载文件
            client.fget_object(bucket_name, obj.object_name, local_file_path)
            print(f"文件 {obj.object_name} 下载成功，保存为 {local_file_path}")
        
        print(f"目录 {prefix} 下载完成")
    except S3Error as err:
        print(f"下载目录失败: {err}")

def upload(path, bucket_name, object_name):
    """
    智能上传文件或目录
    
    Args:
        path: 本地文件或目录路径
        bucket_name: 存储桶名称
        object_name: 对象名称（MinIO中的路径）
    """
    if os.path.isfile(path):
        # 如果object_name以/结尾，则视为目录，将文件放在该目录下
        if object_name.endswith('/'):
            object_name = object_name + os.path.basename(path)
        upload_file(path, bucket_name, object_name)
    elif os.path.isdir(path):
        # 确保object_name以/结尾
        if not object_name.endswith('/'):
            object_name = object_name + '/'
        upload_directory(path, bucket_name, object_name)
    else:
        print(f"错误: {path} 不是有效的文件或目录")

def download(bucket_name, object_name, download_path):
    """
    智能下载文件或目录
    
    Args:
        bucket_name: 存储桶名称
        object_name: 对象名称（MinIO中的路径或目录）
        download_path: 下载目录路径
    """
    # 确保下载路径是目录
    if os.path.exists(download_path) and not os.path.isdir(download_path):
        print(f"错误: 下载路径 {download_path} 必须是目录")
        return
    
    try:
        # 检查object_name是否是目录（通过查看是否有以该前缀开头的对象）
        objects = list(client.list_objects(bucket_name, prefix=object_name, recursive=False))
        
        # 如果只有一个对象且名称与object_name完全匹配，则视为文件
        if len(objects) == 1 and objects[0].object_name == object_name:
            download_file(bucket_name, object_name, download_path)
        else:
            # 确保object_name以/结尾
            if not object_name.endswith('/'):
                object_name = object_name + '/'
            download_directory(bucket_name, object_name, download_path)
    except S3Error as err:
        print(f"下载操作失败: {err}")


if __name__ == '__main__':    
    # 示例用法
    # 上传文件或目录
    file_path = "/code/github/cocox/test/download/coco_data.zip"
    bucket_name = "cvat-cv"
    # object_name = "upload/"
    # upload(file_path, bucket_name, object_name)
    # object_name = "upload/dd/"
    # upload(file_path, bucket_name, object_name)
    object_name = "coco_data.zip"
    upload(file_path, bucket_name, object_name)

    # # 下载文件或目录
    # bucket_name = "test"
    # object_name = "upload/"
    # download(bucket_name, object_name, "./download")
    # # object_name = "upload/dd/"
    # # download(bucket_name, object_name, "./download") # pass
    # object_name = "upload/d1/tt.zip"
    # download("test", "test/coco_data.zip", "./test/download")
