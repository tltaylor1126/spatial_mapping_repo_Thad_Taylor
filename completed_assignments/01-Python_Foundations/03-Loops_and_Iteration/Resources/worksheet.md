# 📝 Worksheet: 04 - Loops and Iteration

Practice and reflect on how loops work in Python.

---

## 🔁 Section 1: For Loops

1. What does `range(5)` produce?

`Answer:` It produces the sequence `0, 1, 2, 3, 4`.

2. Write a `for` loop that prints numbers 1 to 10, but skips 5.

```python
for number in range(1, 11):
    if number == 5:
        continue
    print(number)
```

---

## 🔁 Section 2: While Loops

3. What’s the difference between a `for` loop and a `while` loop?

`Answer:` A `for` loop iterates over a known sequence or range; a `while` loop repeats as long as a condition remains `True`.

4. What happens if a `while` loop's condition never becomes `False`?

`Answer:` It becomes an infinite loop and keeps running until interrupted or the program exits.

---

### ✏️ Task: Countdown with While

```python
# Use a while loop to count down from 5 to 1.
count = 5
while count >= 1:
    print(count)
    count -= 1
```

---

## 📁 Section 3: File Reading and `with`

5. What does the `with` statement do when opening a file?

`Answer:` It manages the file resource and automatically closes the file when the block finishes.

6. How do you loop over each line in a file?

`Answer:` Use a `for` loop directly on the file object, such as `for line in file:`.

---

### ✏️ Task: File Filter

Write code that prints only the lines in a file that contain the word `"error"`.

```python
with open("log.txt", "r", encoding="utf-8") as file:
    for line in file:
        if "error" in line.lower():
            print(line.strip())
```
