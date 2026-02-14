import '../models/course.dart';

class CourseData {
  static List<Course> getCourses() {
    return [
      Course(
        id: 'intro-to-programming',
        title: 'Introduction to Programming',
        description: 'Learn the fundamentals of programming through guided discovery',
        thumbnail: 'assets/images/programming.png',
        difficulty: 'Beginner',
        duration: '4 hours',
        totalLessons: 12,
        modules: [
          Module(
            id: 'mod-1',
            title: 'Getting Started',
            description: 'Understanding what programming is',
            orderIndex: 0,
            chapters: [
              Chapter(
                id: 'ch-1-1',
                title: 'What is Programming?',
                orderIndex: 0,
                lessons: [
                  Lesson(
                    id: 'intro-to-programming-1-1-1',
                    title: 'Understanding Computers',
                    orderIndex: 0,
                    content: '''
# Understanding Computers

Computers are everywhere in our modern world. But what exactly is a computer, and how does it work?

## What is a Computer?

A computer is an electronic device that processes information. Think of it as a very fast calculator that can also store information and follow instructions.

## Key Components

**Hardware** - The physical parts you can touch:
- CPU (Central Processing Unit) - The "brain"
- Memory (RAM) - Short-term storage
- Storage (Hard Drive/SSD) - Long-term storage
- Input devices (keyboard, mouse)
- Output devices (monitor, speakers)

**Software** - The programs and instructions:
- Operating System (Windows, macOS, Linux)
- Applications (browsers, games, editors)

## How Computers Process Information

Computers work with binary - just 0s and 1s. Everything you see on screen, every song you hear, is ultimately represented as patterns of these two digits.

Think about it: How might you represent different things using only "on" and "off" signals?
''',
                    keyPoints: 'Computers process information using hardware and software. All data is binary.',
                    reflectionQuestions: [
                      'Why do you think computers use binary (0s and 1s)?',
                      'What makes a computer different from a calculator?',
                    ],
                  ),
                  Lesson(
                    id: 'intro-to-programming-1-1-2',
                    title: 'What is a Program?',
                    orderIndex: 1,
                    content: '''
# What is a Program?

Now that we understand computers, let's explore what programs are.

## Programs are Instructions

A program is a set of instructions that tells a computer what to do. Just like a recipe tells you how to cook a meal, a program tells a computer how to complete a task.

## Example: Making a Sandwich

If you were to write instructions for making a sandwich, they might look like:

1. Get two slices of bread
2. Open the peanut butter jar
3. Use a knife to spread peanut butter on one slice
4. Open the jelly jar
5. Spread jelly on the other slice
6. Put the two slices together

Notice how specific these instructions need to be. Computers need even MORE specific instructions!

## Why Precision Matters

Computers do exactly what you tell them - nothing more, nothing less. If your instructions are unclear or missing steps, the computer won't "figure it out" like a human might.

## Programming Languages

We write programs using programming languages. These are special languages designed to give precise instructions to computers. Examples include:
- Python
- JavaScript
- Java
- C++

Each language has its own rules and syntax, but they all serve the same purpose: communicating with computers.
''',
                    keyPoints: 'Programs are precise instructions. Computers follow them exactly.',
                    reflectionQuestions: [
                      'Why do you think precision is so important in programming?',
                      'What might happen if you skip a step in your instructions?',
                    ],
                  ),
                ],
              ),
              Chapter(
                id: 'ch-1-2',
                title: 'Your First Code',
                orderIndex: 1,
                lessons: [
                  Lesson(
                    id: 'intro-to-programming-1-2-1',
                    title: 'Hello, World!',
                    orderIndex: 0,
                    content: '''
# Hello, World!

Every programmer's journey begins with the same tradition: writing a program that displays "Hello, World!" on the screen.

## Why "Hello, World"?

This simple program has been the first step for millions of programmers since 1978. It's a way to:
- Verify your programming environment works
- Learn the basic syntax of a language
- Celebrate your first working program!

## In Python

Here's how you write Hello World in Python:

```python
print("Hello, World!")
```

That's it! Just one line. The `print()` function displays text on the screen.

## Breaking It Down

- `print` - A built-in function that outputs text
- `(` and `)` - Parentheses that contain what we want to print
- `"Hello, World!"` - The text we want to display (called a "string")

## Try It Yourself

Think about what would happen if you wrote:
- `print("Your name here")`
- `print(Hello World)` (without quotes)
- `print("Hello, " + "World!")`

What do you predict each would do?
''',
                    keyPoints: 'print() displays text. Strings need quotation marks.',
                    reflectionQuestions: [
                      'Why do you think we need quotation marks around the text?',
                      'What would happen if we forgot the closing parenthesis?',
                    ],
                  ),
                  Lesson(
                    id: 'intro-to-programming-1-2-2',
                    title: 'Variables and Data',
                    orderIndex: 1,
                    content: '''
# Variables and Data

Variables are one of the most fundamental concepts in programming. They let us store and work with information.

## What is a Variable?

Think of a variable as a labeled box where you can store things. The label is the variable's name, and the contents are its value.

```python
name = "Alice"
age = 25
height = 5.6
is_student = True
```

## Types of Data

Different boxes hold different types of things:

**Strings** - Text in quotes
```python
greeting = "Hello"
```

**Integers** - Whole numbers
```python
count = 42
```

**Floats** - Decimal numbers
```python
price = 19.99
```

**Booleans** - True or False
```python
is_active = True
```

## Using Variables

Once you store something in a variable, you can use it:

```python
name = "Alice"
print("Hello, " + name)  # Outputs: Hello, Alice

age = 25
next_year = age + 1  # next_year is now 26
```

## Naming Variables

Good variable names are:
- Descriptive: `user_age` not `x`
- Lowercase with underscores: `total_score`
- Not starting with numbers: `score1` ✓, `1score` ✗
''',
                    keyPoints: 'Variables store data. Different types: strings, integers, floats, booleans.',
                    reflectionQuestions: [
                      'Why is it important to choose good variable names?',
                      'What type of variable would you use to store a temperature?',
                    ],
                  ),
                ],
              ),
            ],
          ),
          Module(
            id: 'mod-2',
            title: 'Control Flow',
            description: 'Making decisions in code',
            orderIndex: 1,
            chapters: [
              Chapter(
                id: 'ch-2-1',
                title: 'Conditionals',
                orderIndex: 0,
                lessons: [
                  Lesson(
                    id: 'intro-to-programming-2-1-1',
                    title: 'If Statements',
                    orderIndex: 0,
                    content: '''
# If Statements

Programs need to make decisions. If statements let your code choose different paths based on conditions.

## The Basic If

```python
temperature = 30

if temperature > 25:
    print("It's hot outside!")
```

The code inside the `if` block only runs when the condition is True.

## If-Else

What if we want to do something different when the condition is False?

```python
temperature = 15

if temperature > 25:
    print("It's hot outside!")
else:
    print("It's not too hot.")
```

## Multiple Conditions with Elif

```python
temperature = 15

if temperature > 30:
    print("It's very hot!")
elif temperature > 20:
    print("It's warm.")
elif temperature > 10:
    print("It's cool.")
else:
    print("It's cold!")
```

## Comparison Operators

- `>` greater than
- `<` less than
- `>=` greater than or equal
- `<=` less than or equal
- `==` equal to (note: two equals signs!)
- `!=` not equal to

## Important: Indentation

Notice the spaces before `print()`? In Python, indentation matters! It tells Python which code belongs inside the if statement.
''',
                    keyPoints: 'If statements make decisions. Use proper indentation. == for comparison.',
                    reflectionQuestions: [
                      'Why do you think Python uses indentation instead of brackets?',
                      'What would happen if you used = instead of == for comparison?',
                    ],
                  ),
                  Lesson(
                    id: 'intro-to-programming-2-1-2',
                    title: 'Logical Operators',
                    orderIndex: 1,
                    content: '''
# Logical Operators

Sometimes you need to check multiple conditions at once. Logical operators help combine conditions.

## AND Operator

Both conditions must be True:

```python
age = 25
has_license = True

if age >= 18 and has_license:
    print("You can drive!")
```

## OR Operator

At least one condition must be True:

```python
day = "Saturday"

if day == "Saturday" or day == "Sunday":
    print("It's the weekend!")
```

## NOT Operator

Reverses a boolean:

```python
is_raining = False

if not is_raining:
    print("No umbrella needed!")
```

## Combining Operators

You can combine multiple operators:

```python
temperature = 22
is_sunny = True

if temperature > 20 and temperature < 30 and is_sunny:
    print("Perfect weather for a walk!")
```

## Truth Tables

Understanding how these work:

| A | B | A and B | A or B |
|---|---|---------|--------|
| T | T | T | T |
| T | F | F | T |
| F | T | F | T |
| F | F | F | F |
''',
                    keyPoints: 'and requires both True. or requires at least one True. not reverses.',
                    reflectionQuestions: [
                      'When would you use AND versus OR in real life decisions?',
                      'How would you check if a number is between 1 and 10?',
                    ],
                  ),
                ],
              ),
              Chapter(
                id: 'ch-2-2',
                title: 'Loops',
                orderIndex: 1,
                lessons: [
                  Lesson(
                    id: 'intro-to-programming-2-2-1',
                    title: 'For Loops',
                    orderIndex: 0,
                    content: '''
# For Loops

Loops let you repeat code multiple times. For loops are great when you know how many times you want to repeat.

## Basic For Loop

```python
for i in range(5):
    print(i)
```

Output:
```
0
1
2
3
4
```

## Understanding range()

`range(5)` creates numbers 0, 1, 2, 3, 4 (not including 5)

You can customize it:
- `range(1, 6)` → 1, 2, 3, 4, 5
- `range(0, 10, 2)` → 0, 2, 4, 6, 8 (counting by 2)

## Looping Through Lists

```python
fruits = ["apple", "banana", "cherry"]

for fruit in fruits:
    print(f"I like {fruit}")
```

Output:
```
I like apple
I like banana
I like cherry
```

## Practical Example

```python
# Calculate sum of numbers 1 to 10
total = 0

for number in range(1, 11):
    total = total + number

print(f"The sum is {total}")  # Output: The sum is 55
```

## Nested Loops

Loops inside loops:

```python
for i in range(3):
    for j in range(3):
        print(f"i={i}, j={j}")
```
''',
                    keyPoints: 'For loops repeat a known number of times. range() generates number sequences.',
                    reflectionQuestions: [
                      'Why does range(5) give 0-4 instead of 1-5?',
                      'How would you print a multiplication table using loops?',
                    ],
                  ),
                  Lesson(
                    id: 'intro-to-programming-2-2-2',
                    title: 'While Loops',
                    orderIndex: 1,
                    content: '''
# While Loops

While loops repeat as long as a condition is True. Use them when you don't know how many repetitions you'll need.

## Basic While Loop

```python
count = 0

while count < 5:
    print(count)
    count = count + 1
```

Output:
```
0
1
2
3
4
```

## Important: Avoid Infinite Loops!

If the condition never becomes False, the loop runs forever:

```python
# DON'T DO THIS - infinite loop!
while True:
    print("This never stops!")
```

Always make sure your loop has a way to end.

## Practical Example: User Input

```python
password = ""

while password != "secret123":
    password = input("Enter password: ")

print("Access granted!")
```

## Break and Continue

**break** - Exit the loop immediately:

```python
for i in range(10):
    if i == 5:
        break  # Stop when i is 5
    print(i)
```

**continue** - Skip to next iteration:

```python
for i in range(5):
    if i == 2:
        continue  # Skip 2
    print(i)
```

## When to Use Which?

- **For loop**: When you know the number of iterations
- **While loop**: When you need to loop until a condition changes
''',
                    keyPoints: 'While loops continue until condition is False. Avoid infinite loops.',
                    reflectionQuestions: [
                      'What could cause an infinite loop?',
                      'When would a while loop be better than a for loop?',
                    ],
                  ),
                ],
              ),
            ],
          ),
          Module(
            id: 'mod-3',
            title: 'Functions',
            description: 'Organizing code into reusable blocks',
            orderIndex: 2,
            chapters: [
              Chapter(
                id: 'ch-3-1',
                title: 'Creating Functions',
                orderIndex: 0,
                lessons: [
                  Lesson(
                    id: 'intro-to-programming-3-1-1',
                    title: 'What are Functions?',
                    orderIndex: 0,
                    content: '''
# What are Functions?

Functions are reusable blocks of code that perform specific tasks. They help organize your code and avoid repetition.

## Why Use Functions?

1. **Reusability** - Write once, use many times
2. **Organization** - Break complex problems into smaller parts
3. **Readability** - Give meaningful names to code blocks
4. **Maintenance** - Fix bugs in one place

## Creating a Function

```python
def greet():
    print("Hello!")
    print("Welcome to programming!")

# Call the function
greet()
```

## Functions with Parameters

Parameters let you pass data into functions:

```python
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")  # Output: Hello, Alice!
greet("Bob")    # Output: Hello, Bob!
```

## Multiple Parameters

```python
def add(a, b):
    result = a + b
    print(f"{a} + {b} = {result}")

add(5, 3)  # Output: 5 + 3 = 8
```

## The DRY Principle

DRY = "Don't Repeat Yourself"

Without functions (repetitive):
```python
print("Hello, Alice!")
print("Hello, Bob!")
print("Hello, Charlie!")
```

With functions (DRY):
```python
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")
greet("Bob")
greet("Charlie")
```
''',
                    keyPoints: 'Functions are reusable code blocks. Parameters pass data to functions.',
                    reflectionQuestions: [
                      'How do functions make code easier to maintain?',
                      'What makes a good function - doing one thing or many things?',
                    ],
                  ),
                  Lesson(
                    id: 'intro-to-programming-3-1-2',
                    title: 'Return Values',
                    orderIndex: 1,
                    content: '''
# Return Values

Functions can send data back using the return statement.

## Basic Return

```python
def add(a, b):
    return a + b

result = add(5, 3)
print(result)  # Output: 8
```

## Return vs Print

**Print** - Displays to screen, nothing is saved
**Return** - Sends value back, can be stored/used

```python
def add_print(a, b):
    print(a + b)  # Just displays

def add_return(a, b):
    return a + b  # Can be used later

# These are different!
x = add_print(5, 3)   # x is None
y = add_return(5, 3)  # y is 8
```

## Multiple Returns

A function can have multiple return statements:

```python
def check_age(age):
    if age >= 18:
        return "Adult"
    else:
        return "Minor"

status = check_age(20)  # "Adult"
```

## Using Returned Values

```python
def square(n):
    return n * n

def sum_of_squares(a, b):
    return square(a) + square(b)

result = sum_of_squares(3, 4)  # 9 + 16 = 25
```

## Returning Multiple Values

Python can return multiple values:

```python
def get_name_parts(full_name):
    parts = full_name.split(" ")
    return parts[0], parts[1]

first, last = get_name_parts("John Doe")
print(first)  # John
print(last)   # Doe
```
''',
                    keyPoints: 'return sends values back. Different from print which only displays.',
                    reflectionQuestions: [
                      'Why would you return a value instead of just printing it?',
                      'What happens after a return statement executes?',
                    ],
                  ),
                ],
              ),
            ],
          ),
        ],
      ),
      Course(
        id: 'critical-thinking',
        title: 'Critical Thinking Skills',
        description: 'Develop analytical and reasoning abilities',
        thumbnail: 'assets/images/thinking.png',
        difficulty: 'Beginner',
        duration: '3 hours',
        totalLessons: 8,
        modules: [
          Module(
            id: 'ct-mod-1',
            title: 'Foundations of Critical Thinking',
            description: 'Understanding what critical thinking means',
            orderIndex: 0,
            chapters: [
              Chapter(
                id: 'ct-ch-1-1',
                title: 'What is Critical Thinking?',
                orderIndex: 0,
                lessons: [
                  Lesson(
                    id: 'critical-thinking-1-1-1',
                    title: 'Introduction to Critical Thinking',
                    orderIndex: 0,
                    content: '''
# Introduction to Critical Thinking

Critical thinking is a skill that helps you analyze information, make better decisions, and solve problems effectively.

## What is Critical Thinking?

Critical thinking is the ability to:
- Analyze information objectively
- Question assumptions
- Evaluate evidence
- Form reasoned judgments
- Consider multiple perspectives

## Why Does It Matter?

In today's world, we're bombarded with information. Critical thinking helps you:
- Distinguish facts from opinions
- Identify reliable sources
- Make informed decisions
- Solve complex problems
- Avoid being manipulated

## The Socratic Method

This course uses the Socratic method - learning through questioning. Instead of being told answers, you'll discover them by:
- Examining your beliefs
- Questioning assumptions
- Following logical reasoning
- Arriving at deeper understanding

## Key Questions Critical Thinkers Ask

1. What is the source of this information?
2. What evidence supports this claim?
3. Are there alternative explanations?
4. What assumptions am I making?
5. What are the implications?
''',
                    keyPoints: 'Critical thinking is analyzing information objectively and questioning assumptions.',
                    reflectionQuestions: [
                      'How do you currently evaluate whether information is trustworthy?',
                      'Can you think of a time when questioning an assumption led to a better outcome?',
                    ],
                  ),
                  Lesson(
                    id: 'critical-thinking-1-1-2',
                    title: 'Cognitive Biases',
                    orderIndex: 1,
                    content: '''
# Cognitive Biases

Our brains take shortcuts that can lead to errors in thinking. Understanding these biases helps us think more clearly.

## What are Cognitive Biases?

Cognitive biases are systematic patterns of deviation from rational judgment. They're mental shortcuts that sometimes mislead us.

## Common Biases

**Confirmation Bias**
We tend to seek information that confirms what we already believe and ignore contradictory evidence.

*Example: Only reading news sources that align with your political views.*

**Anchoring Bias**
We rely too heavily on the first piece of information we receive.

*Example: A \$100 shirt seems cheap if you first saw one priced at \$500.*

**Availability Heuristic**
We judge probability by how easily examples come to mind.

*Example: Fearing plane crashes more than car accidents, even though cars are statistically more dangerous.*

**Bandwagon Effect**
We tend to believe things because many other people believe them.

*Example: Buying a product because it's popular, not because it's best for your needs.*

## Overcoming Biases

1. Be aware they exist
2. Actively seek opposing viewpoints
3. Ask "What evidence would change my mind?"
4. Slow down important decisions
5. Consider base rates and statistics
''',
                    keyPoints: 'Cognitive biases are mental shortcuts that can mislead us. Awareness helps overcome them.',
                    reflectionQuestions: [
                      'Which cognitive bias do you think affects you most often?',
                      'How might confirmation bias affect research on controversial topics?',
                    ],
                  ),
                ],
              ),
            ],
          ),
        ],
      ),
    ];
  }
}
