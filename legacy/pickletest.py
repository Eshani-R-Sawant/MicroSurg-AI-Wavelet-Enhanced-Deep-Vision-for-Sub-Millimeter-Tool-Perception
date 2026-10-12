import pickle

# 🔥 IMPORTANT: set correct path to your .p file
PICKLE_PATH = "./SurgMLLMBench/SurgMLLMBench/EndoVis2018/train/endovis2018_segmentation_train.p"

with open(PICKLE_PATH, "rb") as f:
    data = pickle.load(f)

print("Type of data:", type(data))
print("Length:", len(data))

print("\nFirst element type:", type(data[0]))
print("\nFirst element content:\n", data[0])
