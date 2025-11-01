def read_libsvm_file(file_path):
    """
    Reads a LIBSVM-formatted text file to extract labels and features data.

    Args:
        file_path (str): The path to the LIBSVM format file.

    Returns:
        tuple: (labels, features_list)
               labels (list): List of labels (float or int) for each instance.
               features_list (list): List of feature dictionaries, where each
                                     dictionary is formatted as {index (int): value (float)}.
    """
    labels = []
    features_list = []

    try:
        with open(file_path, "r") as f:
            for line in f:
                # Remove leading/trailing whitespace and newline characters
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                # 1. Separate label and feature data
                parts = line.split(" ", 1)
                if len(parts) < 1:
                    continue

                label_str = parts[0]
                features_str = parts[1] if len(parts) > 1 else ""

                # Store the label (convert to float/int)
                try:
                    labels.append(float(label_str))
                except ValueError:
                    print(f"Warning: Could not convert label '{label_str}' to a number. Skipping this line.")
                    continue

                # 2. Extract and store features (index:value)
                feature_dict = {}
                feature_pairs = features_str.split()  # Separate feature pairs by space

                for pair in feature_pairs:
                    if ":" in pair:
                        index_str, value_str = pair.split(":", 1)
                        try:
                            # Use the LIBSVM index as is (it's 1-based)
                            index = int(index_str)
                            value = float(value_str)
                            feature_dict[index] = value
                        except ValueError:
                            # Handle cases where feature index or value is not a number
                            print(f"Warning: Invalid feature pair '{pair}' detected and ignored.")
                            continue

                features_list.append(feature_dict)

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return [], []
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        return [], []

    return labels, features_list


# --- Usage Example ---

# 1. Create a sample LIBSVM data file (virtual file)
# Label(1), (Index 3: Value 0.8), (Index 10: Value 0.1)
# Label(-1), (Index 1: Value 0.5), (Index 3: Value 0.2), (Index 10: Value 0.9)
example_file_path = "example_libsvm_data.txt"
with open(example_file_path, "w") as f:
    f.write("1 3:0.8 10:0.1\n")
    f.write("-1 1:0.5 3:0.2 10:0.9\n")
    f.write("2 5:1.0\n")  # Indices 1, 2, 3, 4, 6.. are 0
    f.write("\n")  # Test empty line

# 2. Call the function
labels, features = read_libsvm_file(example_file_path)

# 3. Output the results
print("\n--- Loaded Data Results ---")
print(f"Total instances: {len(labels)}")
print("\nLabels:")
print(labels)

print("\nFeatures (index:value):")
for i, f_dict in enumerate(features):
    print(f"Instance {i + 1} features: {f_dict}")
