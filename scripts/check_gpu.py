import tensorflow as tf

# 사용 가능한 GPU 목록 출력
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"사용 가능한 GPU: {gpus}")
    for gpu in gpus:
        gpu_details = tf.config.experimental.get_device_details(gpu)
        print(f"GPU 상세 정보: {gpu_details}")
else:
    print("사용 가능한 GPU가 없습니다.")
