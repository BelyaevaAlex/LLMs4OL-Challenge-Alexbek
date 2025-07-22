# 🏗️ Cross-Attention Model Architecture

## 📐 Data Structure

### **Relation Direction**
In our data we have explicit parent-child relationships:
```json
{
  "ID": "TR_7e0265b4", 
  "parent": "measured property",
  "child": "area"
}
```

### **Architectural Solution**
```
vectors_1 = children (queries)   - who is looking for parents
vectors_2 = parents (keys/values) - among whom we search for parents
matrix[i, j] = 1 if child_i has parent_j
```

### **Matrix Structure**
```
           parents →
children ↓  [p1] [p2] [p3]
    [c1]     1    0    1     # c1 has parents p1, p3
    [c2]     0    1    0     # c2 has parent p2  
    [c3]     1    1    0     # c3 has parents p1, p2
```

## ⚡ Relation Matrix Optimization

### **Problem with Old Algorithm**
```python
# O(n*m) - slow full matrix traversal
for i, child_id in enumerate(ids_1):
    for j, parent_id in enumerate(ids_2):
        if (child_id, parent_id) in relations_set:
            matrix[i, j] = 1.0
```

### **New Optimized Algorithm**
```python
# O(n*k) - fast search only for needed positions
# where k is the average number of parents per child

# 1. Create index child -> {parent1, parent2, ...}
child_to_parents = {
    child_id: {parent_id1, parent_id2, ...}
}

# 2. For each child find intersection with current batch
for i, child_id in enumerate(ids_1):
    child_parents = child_to_parents[child_id]
    for parent_id in child_parents:
        if parent_id in current_batch_ids_2:
            j = parent_id_to_col[parent_id]
            matrix[i, j] = 1.0
```

### **Advantages**
- **Speed**: O(n*k) instead of O(n*m)
- **Efficiency**: Don't traverse entire matrix, set 1 only where needed
- **Clarity**: Logic "for each child find its parents in the batch"

## 🎯 Usage Examples

### **Balanced Strategy**
```python
dataset = CrossAttentionDataset(
    entities_path="entities.json",
    relations_path="relations.json", 
    batch_size_1=16,        # children
    batch_size_2=16,        # parents
    sampling_strategy="balanced",
    positive_ratio=1.0      # maximum positive pairs
)

# Get batch
children_vectors, parents_vectors, matrix, dataset_name = dataset[0]

# matrix.shape = (16, 16) - [children, parents]
# matrix[i, j] = 1 if child_i has parent_j
```

### **Optimization Results**
- **6.3%** positive pairs (instead of 1.8% before)
- **3.5 times** more positive examples
- **Fast** relation matrix creation

## 🔧 Technical Details

### **Indexing**
- `child_to_parents[dataset_name][child_id]` → `{parent_id1, parent_id2, ...}`
- Created once during dataset initialization
- Used for fast relation lookup

### **Sampling Strategies**
- **random**: Random entity selection
- **balanced**: Maximize positive pairs in matrix

### **Operation Modes**
- **single**: Single dataset
- **weighted**: Weighted selection based on number of relations 