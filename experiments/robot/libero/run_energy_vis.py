import tensorflow as tf

# 读取 tfrecord
tfrecord_path = "/work1/aiginternal/yuhang/modified_libero_rlds/1.0.0/liber_o10-train.tfrecord-00000-of-00032"
raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

for raw_record in raw_dataset.take(1):  # 先看一个样本
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)
