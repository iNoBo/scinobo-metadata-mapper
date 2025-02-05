from collections import defaultdict, Counter
import json

# Data structures to store information
alternate_names_with_fullnames = defaultdict(list)
data = []

# Read the JSON file line by line
with open("publication_venues_v1.0.json", "r") as fp:
    for line in fp:
        data.append(json.loads(line))

# Process each entry
for d in data:
    alt_set = set(d["alternate_names"])  # Avoid in-row duplicates
    for alt_name in alt_set:
        alternate_names_with_fullnames[alt_name].append(d["name"])

# Count the occurrences of each alternate name
#name_to_count = Counter(alternate_names_with_fullnames.keys())

duplicates = {
    name: fullnames
    for name, fullnames in alternate_names_with_fullnames.items()
    if len(fullnames) > 1
}

# Write the duplicates to a file
output_file = "duplicates_with_fullnames.json"
with open(output_file, "w") as fp:
    json.dump(duplicates, fp, indent=4)

print(f"Duplicates written to {output_file}")
print(f"Number of alternate name duplicates: {len(duplicates)} out of {len(data)} entries.")