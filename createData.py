book_data = open("trainingData.txt", "r").read()

# Split by character position — never shuffle sequential text
n = len(book_data)
train_end = int(0.80 * n)
val_end   = int(0.90 * n)

train_data = book_data[:train_end]
val_data   = book_data[train_end:val_end]
test_data  = book_data[val_end:]

print(f"Total chars : {n:,}")
print(f"Train       : {len(train_data):,}  ({len(train_data)/n:.1%})")
print(f"Validation  : {len(val_data):,}   ({len(val_data)/n:.1%})")
print(f"Test        : {len(test_data):,}   ({len(test_data)/n:.1%})")


with open("train.txt", "w") as f:
    f.write(train_data)

with open("val.txt", "w") as f:
    f.write(val_data)

with open("test.txt", "w") as f:
    f.write(test_data)