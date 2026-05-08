# 📝 Worksheet: 02 - Working with Data

Use this worksheet to review and reinforce your understanding of Python data containers.

---

## 🧠 Section 1: Lists

1. What method adds an item to the end of a list?  
   `Answer:` `.append()`

2. How can you remove an item from a list by value?  
   `Answer:` `list.remove(value)` removes the first matching item by value.

3. What’s the result of this code?

```python
nums = [2, 4, 6]
nums.append(8)
print(nums)
```

   `Answer:` `[2, 4, 6, 8]`

---

### ✏️ Task: List Practice

```python
foods = ["tacos", "sushi", "pasta"]
foods.append("pizza")
foods.remove("sushi")
print(foods)
```

---

## 🔒 Section 2: Tuples

4. What is a key difference between a list and a tuple?  
   `Answer:` Lists are mutable; tuples are immutable.

5. Can you change the contents of a tuple once it is created? Why or why not?  
   `Answer:` No. Tuples cannot be changed in place after creation.

---

### ✏️ Task: Tuple Practice

```python
numbers = (7, 11, 26)
a, b, c = numbers
print(a, b, c)
```

---

## 🔑 Section 3: Dictionaries

6. What does the `.get()` method do differently from accessing a key directly?  
   `Answer:` `.get()` returns a default value instead of raising `KeyError` when the key is missing.

7. How do you loop through both keys and values in a dictionary?  
   `Answer:` Use `for key, value in dictionary.items():`.

---

### ✏️ Task: Dictionary Practice

```python
person = {"name": "Thad", "age": 21, "hobby": "mapping"}
for key, value in person.items():
    print(f"{key}: {value}")
```

---

## 🧾 Submit Checklist

- [x] I practiced creating and modifying lists.
- [x] I understand how tuples are different from lists.
- [x] I accessed and looped through dictionary items.
