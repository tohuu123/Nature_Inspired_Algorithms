# Nature Inspired Algorithms

This project implements various nature-inspired optimization algorithms

## Installation

Install the required dependencies:

```bash
pip install -r requirement.txt
```

## Testing

### Test Genetic Algorithm (GA)

The GA algorithm is designed for discrete optimization:

```bash
cd algorithms/biology/GA
python GA.py
```

### Test Differential Evolution (DE)

The DE algorithm can be tested on various continuous optimization functions:

```bash
cd algorithms/biology/DE
python DE.py
```

## Project Structure

```
Nature_Inspired_Algorithms/
├── algorithms/
│   ├── biology/
│   │   ├── DE/
│   │   │   └── DE.py          # Differential Evolution
│   │   └── GA/
│   │       └── GA.py          # Genetic Algorithm
│   ├── classical/
│   ├── evolutionary/
│   ├── human/
│   ├── physics/
│   └── testing/
│       └── continous/
│           └── function.py    # Test functions (sphere, rastrigin, etc.)
├── requirement.txt
└── README.md
```


