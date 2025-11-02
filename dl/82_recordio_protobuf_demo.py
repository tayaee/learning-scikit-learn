import struct
import io
import os
import numpy as np

# Import the Protobuf module generated in Step 2.
from sample_pb2 import TrainRecord

# --- 1. File Writing Function (Write) ---


def write_recordio_protobuf(filename, data_list):
    """
    Serializes data using Protobuf, prefixes with the RecordIO header, and writes to a file.
    """
    print(f"--- Starting file writing: {filename} ---")

    # Open file in binary write mode ('wb')
    with open(filename, "wb") as f:
        for features, label in data_list:
            # 1. Create Protobuf object and assign data
            record = TrainRecord()
            # Assign NumPy array features by converting to a list
            record.features.extend(features.tolist())
            record.label = label

            # 2. Serialize Protobuf to a binary byte string (Protobuf operation)
            serialized_data = record.SerializeToString()

            # 3. Create the RecordIO header (4-byte length information) (RecordIO operation)
            # 'I' denotes a 4-byte unsigned integer.
            data_length = len(serialized_data)
            length_prefix = struct.pack("I", data_length)

            # 4. Write the RecordIO header and the serialized data to the file
            f.write(length_prefix)
            f.write(serialized_data)

            print(f"  > Record written. Length: {data_length} bytes")

    print("--- File writing completed ---")


# --- 2. File Reading Function (Deserialization) ---


def read_recordio_protobuf(filename):
    """
    Reads the RecordIO header, then deserializes the Protobuf message.
    """
    print(f"\n--- Starting file reading: {filename} ---")
    read_data = []

    # Open file in binary read mode ('rb')
    with open(filename, "rb") as f:
        while True:
            # 1. Read the RecordIO header (4 bytes for length)
            # Reads exactly 4 bytes from the file.
            length_prefix = f.read(4)

            if not length_prefix:
                # Reached End of File (EOF)
                break

            # 2. Deserialize the 4 bytes into an unsigned integer (Record length) (RecordIO operation)
            data_length = struct.unpack("I", length_prefix)[0]

            # 3. Read the serialized data bytes based on the length information
            serialized_data = f.read(data_length)

            if len(serialized_data) < data_length:
                # Error: Data is incomplete (real-world error handling required)
                print("Error: Data is incomplete.")
                break

            # 4. Create a Protobuf object and parse the bytes to restore the record (Protobuf operation)
            record = TrainRecord()
            record.ParseFromString(serialized_data)

            # 5. Store the restored data
            read_data.append({"features": np.array(record.features), "label": record.label})

            print(f"  > Record read. Label: {record.label}")

    print("--- File reading completed ---")
    return read_data


# --- Execution Example ---

# 1. Generate sample training data (using NumPy arrays)
sample_data = [
    (np.array([1.0, 2.5, 3.1]), 0.0),
    (np.array([4.2, 5.5, 6.7]), 1.0),
    (np.array([7.8, 8.9, 9.3]), 0.0),
]

output_file = "sagemaker_train_data.recordio_pb"

# Write the file
write_recordio_protobuf(output_file, sample_data)

# Read the file
loaded_data = read_recordio_protobuf(output_file)

# Verify the result
print("\n--- Verifying restored data ---")
for item in loaded_data:
    print(f"Features: {item['features']}, Label: {item['label']}")

# Clean up the file
os.remove(output_file)
