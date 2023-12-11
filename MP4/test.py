# Original tuple
original_tuple = (1, "abs", 3, 4, "5")

# Element to exclude
element_to_exclude = 4

# Create a new tuple by filtering out the element
new_tuple = tuple(item for item in original_tuple if item != element_to_exclude)

print(new_tuple)