# Python_Portfolio_BStevenson
My portfolio of learned content from Dr. V;s class


## Using Jupyter Notebooks Pts 1-2

```python
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'darkgrid')
```


```python
df = pd.read_csv('/home/student/Desktop/classroom/myfiles/notebooks/fortune500.csv')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Rank</th>
      <th>Company</th>
      <th>Revenue (in millions)</th>
      <th>Profit (in millions)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1955</td>
      <td>1</td>
      <td>General Motors</td>
      <td>9823.5</td>
      <td>806</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1955</td>
      <td>2</td>
      <td>Exxon Mobil</td>
      <td>5661.4</td>
      <td>584.8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1955</td>
      <td>3</td>
      <td>U.S. Steel</td>
      <td>3250.4</td>
      <td>195.4</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1955</td>
      <td>4</td>
      <td>General Electric</td>
      <td>2959.1</td>
      <td>212.6</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1955</td>
      <td>5</td>
      <td>Esmark</td>
      <td>2510.8</td>
      <td>19.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Rank</th>
      <th>Company</th>
      <th>Revenue (in millions)</th>
      <th>Profit (in millions)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>25495</td>
      <td>2005</td>
      <td>496</td>
      <td>Wm. Wrigley Jr.</td>
      <td>3648.6</td>
      <td>493</td>
    </tr>
    <tr>
      <td>25496</td>
      <td>2005</td>
      <td>497</td>
      <td>Peabody Energy</td>
      <td>3631.6</td>
      <td>175.4</td>
    </tr>
    <tr>
      <td>25497</td>
      <td>2005</td>
      <td>498</td>
      <td>Wendy's International</td>
      <td>3630.4</td>
      <td>57.8</td>
    </tr>
    <tr>
      <td>25498</td>
      <td>2005</td>
      <td>499</td>
      <td>Kindred Healthcare</td>
      <td>3616.6</td>
      <td>70.6</td>
    </tr>
    <tr>
      <td>25499</td>
      <td>2005</td>
      <td>500</td>
      <td>Cincinnati Financial</td>
      <td>3614.0</td>
      <td>584</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns = ['year', 'rank', 'company', 'revenue', 'profit']
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>rank</th>
      <th>company</th>
      <th>revenue</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1955</td>
      <td>1</td>
      <td>General Motors</td>
      <td>9823.5</td>
      <td>806</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1955</td>
      <td>2</td>
      <td>Exxon Mobil</td>
      <td>5661.4</td>
      <td>584.8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1955</td>
      <td>3</td>
      <td>U.S. Steel</td>
      <td>3250.4</td>
      <td>195.4</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1955</td>
      <td>4</td>
      <td>General Electric</td>
      <td>2959.1</td>
      <td>212.6</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1955</td>
      <td>5</td>
      <td>Esmark</td>
      <td>2510.8</td>
      <td>19.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(df)
```




    25500




```python
df.dtypes
```




    year         int64
    rank         int64
    company     object
    revenue    float64
    profit      object
    dtype: object




```python
non_numeric_profits = df.profit.str.contains('[^0-9.-]')
df.loc[non_numeric_profits].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>rank</th>
      <th>company</th>
      <th>revenue</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>228</td>
      <td>1955</td>
      <td>229</td>
      <td>Norton</td>
      <td>135.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>290</td>
      <td>1955</td>
      <td>291</td>
      <td>Schlitz Brewing</td>
      <td>100.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>294</td>
      <td>1955</td>
      <td>295</td>
      <td>Pacific Vegetable Oil</td>
      <td>97.9</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>296</td>
      <td>1955</td>
      <td>297</td>
      <td>Liebmann Breweries</td>
      <td>96.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>352</td>
      <td>1955</td>
      <td>353</td>
      <td>Minneapolis-Moline</td>
      <td>77.4</td>
      <td>N.A.</td>
    </tr>
  </tbody>
</table>
</div>




```python
set(df.profit[non_numeric_profits])
```




    {'N.A.'}




```python
len(df.profit[non_numeric_profits])
```




    369




```python
bin_sizes, _, _ = plt.hist(df.year[non_numeric_profits], bins= range(1955, 2006))
```


![png](output_11_0.png)



```python
df = df.loc[~non_numeric_profits]
df.profit = df.profit.apply(pd.to_numeric)
```


```python
len(df)
```




    25131




```python
df.dtypes
```




    year         int64
    rank         int64
    company     object
    revenue    float64
    profit     float64
    dtype: object




```python
group_by_year = df.loc[:, ['year', 'revenue', 'profit']].groupby('year')
avgs = group_by_year.mean()
x = avgs.index
y1 = avgs.profit
def plot(x, y, ax, title, y_label):
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.plot(x, y)
    ax.margins(x = 0, y = 0)
```


```python
fig, ax = plt.subplots()
plot(x, y1, ax, 'Increase in mean Fortune 500 company profits from 1955 to 2005', 'Profit (millions)')
```


![png](output_16_0.png)



```python
y2 = avgs.revenue
fig, ax = plt.subplots()
plot(x, y2, ax, 'Increase in mean Fortune 500 company revenues from 1955 50 2005', 'Revenue (millions)')
```


![png](output_17_0.png)



```python
def plot_with_std(x, y, stds, ax, title, y_label):
    ax. fill_between(x, y - stds, y + stds, alpha = 0.2)
    plot(x, y, ax, title, y_label)
fig, (ax1, ax2) = plt.subplots(ncols= 2)
title = 'Increase in mean ans std Fortune 500 company %s from 1955 to 2005'
stds1 = group_by_year.std().profit.values
stds2 = group_by_year.std().revenue.values
plot_with_std(x, y1.values, stds1, ax1, title % 'profits', 'Profit (millions)')
plot_with_std(x, y2.values, stds2, ax2, title % 'revenues', 'Revenue (millions)')
fig.set_size_inches(14,4)
fig.tight_layout()
```


![png](output_18_0.png)



```python

```




## Using Loops

```python
odds = [1,2,3,5,7]
```


```python
print(odds[0])
print(odds[1])
print(odds[2])
print(odds[3])
```

    1
    2
    3
    5



```python
odds = [1,3,5]
print(odds[0])
print(odds[1])
print(odds[2])
print(odds[3])
```

    1
    3
    5



    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-3-01ba67d8a9e5> in <module>
          3 print(odds[1])
          4 print(odds[2])
    ----> 5 print(odds[3])
    

    IndexError: list index out of range



```python
odds = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

for num in odds:
    print(num)
```

    1
    3
    5
    7
    9
    11
    13
    15
    17
    19



```python
length = 0
names = ['Curie', 'Darwin', 'Turing']
for vlaue in names:
    length = length + 1
print('There are', length, 'names in the list.')
```

    There are 3 names in the list.



```python
name = 'Rosalind'
for name in ['Curie', 'Darwin', 'Turing']:
    print(name)
print('after the loop, name is:', name)
```

    Curie
    Darwin
    Turing
    after the loop, name is: Turing



```python
print(len([0,1,2,3]))
```

    4



```python
name = ['Curie', 'Darwin', 'Turing']

print(len(name))
```

    3



```python

```




## Python Fundamentals

```python
#Any python interpretor can be used as a calculator
3 + 5 * 4
```




    23




```python
# Let's save a value to a varuable
weight_kg = 60
```


```python
print(weight_kg)
```

    60



```python
# Weight0 = valid
# 0weight = invalid
# weight and Weight are different
```


```python
# Types of data
# There are three common types of dat
# Interger numbers
# Floating point numbers
# Strings
```


```python
# Floating point
weight_kg = 60.3
```


```python
# String of letters
patient_name = "Jon Smith"
```


```python
# String of numbers
patient_id = '001'
```


```python
print(patient_id)
```

    0.001



```python
# Use variables on python

weight_lb = 2.2 * weight_kg

print(weight_lb)
```

    132.66



```python
# Let's add a prefix to the patient id

patient_id = 'inflam_' + patient_id

print(patient_id)
```

    inflam_001



```python
# Let's combine print statements

print(patient_id, 'weight in kilograms:', weight_kg)
```

    inflam_001 weight in kilograms: 60.3



```python
# We can call a function inside of another function

print(type(60.3))

print(type(patient_id))
```

    <class 'float'>
    <class 'str'>



```python
# We can also do calculations inside the print function

print('weight in pounds:', 2.2 * weight_kg)
```

    weight in pounds: 132.66



```python
print(weight_kg)
```

    60.3



```python
weight_kg = 65.0
print('weight in kilograms is now:', weight_kg)
```

    weight in kilograms is now: 65.0



```python

```




## Translating RNA to Protein

```python
# Prompt the user to enter the input file name

input_file_name = input('Enter the name of the input RNA file:')
```

    Enter the name of the input RNA file: Ubiquitin



```python
# Open the input file and read the RNA sequence

with open(input_file_name, 'r') as input_file:
    rna_sequence = input_file.read().strip()
```


```python
# Define the codon table

codon_table = {
    'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
    'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACU': 'I', 'ACC': 'I', 'ACA': 'I', 'ACG': 'I',
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',
    'CAU': 'M', 'CAC': 'M', 'CAA': 'Q', 'CAG': 'Q', 
    'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
    
}
```


```python
# Translate RNA to protein

protein_sequence = ''
for i in range(0, len(rna_sequence), 3):
    codon = rna_sequence[i:i+3]
    if len(codon) == 3:
        amino_acid = codon_table[codon]

    protein_sequence += amino_acid
```


```python
# Prompt the user to enter the name of the output file

output_file_name = input('Enter the name of the output file: ')
```

    Enter the name of the output file:  Ubuquitin_Protein.txt



```python
# Save the protein sequence to a text file

with open(output_file_name, 'w') as output_file:
    output_file.write(protein_sequence)
    print(f'The protein sequence has been saved to {output_file_name}')
```

    The protein sequence has been saved to Ubuquitin_Protein.txt



```python
print(protein_sequence)
```

    MSDEKKGGEIEMINLKVLGQDNAVVQFKIKKMIPLRKLMNAYCDRAGLSMQVVRFRFDGQPINENDIPISLEMEEGDIIEVYQQQIGGAP*MSDEKKGGEIEMINLKVLGQDNAVVQFKIKKMIPLRKLMNAYCDRAGLSMQVVRFRFDGQPINENDIPISLEMEEGDIIEVYQQQIGGAP*



```python

```




## Transcribing DNA to RNA

```python
# Prompt user to enter the input FASTA file name

input_file_name = input('Enter the name of the input FASTA file:')
```

    Enter the name of the input FASTA file: SUMO.txt



```python
# Open the input file and read the dna sequence

with open(input_file_name, 'r') as input_file:
    dna_sequence = ''
    for line in input_file:
        if line.startswith('>'):
            continue
        dna_sequence += line.strip()
```


```python
# Transcribe the DNA to RNA

rna_sequence = ''
for nucleotide in dna_sequence:
    if nucleotide == 'T':
        rna_sequence += 'U'
    else:
        rna_sequence += nucleotide
```


```python
# Prompt the user to enter the output file name

output_file_name = input('Enter the name of the output file:')
```

    Enter the name of the output file: Ubiquitin



```python
# Save the RNA sequence to a text file

with open(output_file_name, 'w') as output_file:
    output_file.write(rna_sequence)
    print(f'The RNA sequence has been saved to: {output_file_name}')
```

    The RNA sequence has been saved to: Ubiquitin



```python
print(rna_sequence)
```

    AUGUCUGACGAAAAGAAGGGAGGUGAGACCGAGCACAUCAACCUGAAGGUCCUCGGCCAGGACAACGCCGUCGUCCAGUUCAAGAUCAAGAAGCACACACCCUUGAGGAAGCUGAUGAACGCCUACUGCGACCGUGCCGGACUCUCCAUGCAGGUGGUGCGCUUCCGUUUCGACGGACAGCCCAUCAACGAGAACGACACUCCGACCUCGCUGGAGAUGGAGGAGGGCGACACCAUCGAGGUUUACCAGCAGCAGACUGGUGGCGCUCCAUAAAUGUCUGACGAAAAGAAGGGAGGUGAGACCGAGCACAUCAACCUGAAGGUCCUCGGCCAGGACAACGCCGUCGUCCAGUUCAAGAUCAAGAAGCACACACCCUUGAGGAAGCUGAUGAACGCCUACUGCGACCGUGCCGGACUCUCCAUGCAGGUGGUGCGCUUCCGUUUCGACGGACAGCCCAUCAACGAGAACGACACUCCGACCUCGCUGGAGAUGGAGGAGGGCGACACCAUCGAGGUUUACCAGCAGCAGACUGGUGGCGCUCCAUAA



```python

```




## Defensive Programming

```python
# In this lesson we will make functions that check themselves for potential errors
```


```python
numbers = [1.5, 2.3, 0.7, 0.001, 4.4]
total = 0.0
for num in numbers:
    assert num > 0.0, 'Data should only contain positive values'
    total += num
print('Total is:', total)
```

    Total is: 8.901



```python
def normalize_rectangle(rect):
    '''Normalizes a rectangle so that it is at the origin and 1.0 units long on its longest axis.
    input should be of the format (x0, y0, x1, y1,).
    (x0, y0,) and (x1, y1) define the lower left and upper right corners of the rectangle respectively.'''
    assert len(rect) == 4, 'Rectangles must contain 4 coordinated'
    x0, y0, x1, y1 = rect
    assert x0 < x1, 'Invalid X coordinates'
    assert y0 < y1, 'Invalid Y coordinates'
    
    dx = x1 - x0
    dy = y1 - y0
    if dx > dy:
        scaled = dy / dx
        upper_x, upper_y = scaled, 1.0
    else:
        scaled = dx / dy
        upper_x, upper_y = scaled, 1.0
        
    assert 0 < upper_x <= 1.0, 'Calculated upper X coordinate invalid'
    assert 0 < upper_y <= 1.0, 'Calculated upper Y coordinate invalid'
    
    return(0, 0, upper_x, upper_y)
```


```python
print(normalize_rectangle((0.0, 1.0, 2.0)))
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-22-a81b6ed7619a> in <module>
    ----> 1 print(normalize_rectangle((0.0, 1.0, 2.0)))
    

    <ipython-input-21-37ac78efbd63> in normalize_rectangle(rect)
          3     input should be of the format (x0, y0, x1, y1,).
          4     (x0, y0,) and (x1, y1) define the lower left and upper right corners of the rectangle respectively.'''
    ----> 5     assert len(rect) == 4, 'Rectangles must contain 4 coordinated'
          6     x0, y0, x1, y1 = rect
          7     assert x0 < x1, 'Invalid X coordinates'


    AssertionError: Rectangles must contain 4 coordinated



```python
print(normalize_rectangle((4.0, 2.0, 1.0, 5.0)))
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-23-5e28a32bada1> in <module>
    ----> 1 print(normalize_rectangle((4.0, 2.0, 1.0, 5.0)))
    

    <ipython-input-21-37ac78efbd63> in normalize_rectangle(rect)
          5     assert len(rect) == 4, 'Rectangles must contain 4 coordinated'
          6     x0, y0, x1, y1 = rect
    ----> 7     assert x0 < x1, 'Invalid X coordinates'
          8     assert y0 < y1, 'Invalid Y coordinates'
          9 


    AssertionError: Invalid X coordinates



```python
print(normalize_rectangle((0.0, 0.0, 1.0, 5.0)))
```

    (0, 0, 0.2, 1.0)



```python
print(normalize_rectangle((0.0, 0.0, 5.0, 1.0)))
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-25-1337bef8f4bf> in <module>
    ----> 1 print(normalize_rectangle((0.0, 0.0, 5.0, 1.0)))
    

    <ipython-input-21-37ac78efbd63> in normalize_rectangle(rect)
         17         upper_x, upper_y = scaled, 1.0
         18 
    ---> 19     assert 0 < upper_x <= 1.0, 'Calculated upper X coordinate invalid'
         20     assert 0 < upper_y <= 1.0, 'Calculated upper Y coordinate invalid'
         21 


    AssertionError: Calculated upper X coordinate invalid



```python

```




## Functions Pt 1

```python
farenheit_val = 99
celsius_val = ((farenheit_val - 32) * (5 / 9))

print(celsius_val)
```

    37.22222222222222



```python
farenheit_val2 = 43
celsius_val2 = ((farenheit_val2 - 32) * (5 / 9))

print(celsius_val2)
```

    6.111111111111112



```python
# Use the def command to define a function

def explicit_fahr_to_celsius(temp):
    # Assign the converted value to a variable
    converted = ((temp - 32) * (5 / 9))
    # Return the values of the new variable
    return converted
```


```python
def fahr_to_celsius(temp):
    # Return values more efficiently than the above code, using the return function without creating a new variable
    return ((temp- 32) * (5 / 9))

```


```python
fahr_to_celsius(32)
```




    0.0




```python
explicit_fahr_to_celsius(32)
```




    0.0




```python
print('Freezing point of water:', fahr_to_celsius(32), 'C')
print('Boiling point of water:', fahr_to_celsius(212), 'C')
```

    Freezing point of water: 0.0 C
    Boiling point of water: 100.0 C



```python
def celsius_to_kelvin(temp_c):
    return temp_c + 273.15

print('Freezing point of water in Kelvin', celsius_to_kelvin(0.))
```

    Freezing point of water in Kelvin 273.15



```python
def fahr_to_kelvin(temp_f):
    temp_c = fahr_to_celsius(temp_f)
    temp_k = celsius_to_kelvin(temp_c)
    return temp_k

print('Boiling point of water in Kelvin', fahr_to_kelvin(212.))
```

    Boiling point of water in Kelvin 373.15



```python
# print('Again, temperature in Kelvin was:', temp_k)

# This will fail because temp_k is temporary, unless we define it outside of a function
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-22-eed2471d229b> in <module>
    ----> 1 print('Again, temperature in Kelvin was:', temp_k)
    

    NameError: name 'temp_k' is not defined



```python
temp_kelvin = fahr_to_kelvin(212.0)
print('Temperature in kelvin was:', temp_kelvin)
```

    Temperature in kelvin was: 373.15



```python
def print_temperatures():
    print('Temperature in Fahrenheit was:', temp_fahr)
    print('Temperature in Kelvin was:', temp_kelvin)
    
temp_fahr = 212.0
temp_kelvin = fahr_to_kelvin(temp_fahr)

print_temperatures()
```

    Temperature in Fahrenheit was: 212.0
    Temperature in Kelvin was: 373.15



```python

```



## Making Choices Pt 2

```python
import numpy
```


```python
data = numpy.loadtxt(fname='inflammation-01.csv', delimiter=',')
```


```python
max_inflammation_0 = numpy.amax(data, axis = 0)[0]
```


```python
# This checks if the maximum values match the specified nnumbers (0) and (20)

max_inflammation_20 = numpy.amax(data, axis = 0)[20]

if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print('Suspicious Maxima Alarm!')
    
elif numpy.sum(numpy.amin(data, axis = 0)) ==0:
    print('Minima add up to zero!')
    
else:
    print('Seems OK!')
```

    Suspicious Maxima Alarm!



```python
data = numpy.loadtxt(fname = 'inflammation-03.csv', delimiter = ',')

max_inflammation_0 = numpy.amax(data, axis = 0)[0]

max_inflammation_20 = numpy.amax(data, axis = 0)[20]

if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print('Suspicious Maxima Alarm!')
    
elif numpy.sum(numpy.amin(data, axis = 0)) ==0:
    print('Minima add up to zero! -> Healthy Participant Alert!')
    
else:
    print('Seems OK!')
```

    Minima add up to zero! -> Healthy Participant Alert!



```python

```



## Making Choices Pt 1

```python
# this lesson explores conditional functions

num = 37
if num > 100:
    print('greater')
else:
    print('not greater')
print('done')
```

    not greater
    done



```python
num = 53
print('before conditional...')
if num > 100:
    print('is greater than 100')
print('...after conditional')
```

    before conditional...
    ...after conditional



```python
num = -3

if num > 0:
    print(num, 'is positive')
elif num == 0:
    print(num, 'is zero')
else:
    print(num, 'is negative')
```

    -3 is negative



```python
if (1 > 0) and (9-1 >= 0):
    print('both parts are true')
else:
    print('at least one part is false')
```

    both parts are true



```python
if (1 > 0) or (9-1 >= 0):
    print('at least one part is true')
else:
    print('both of these are false')
```

    at least one part is true



```python
import numpy
```



## Using Multiple Files

```python
import glob
```


```python
# This function tells us all the files in the directory that start with 'inflamation' and end with '.csv'

print(glob.glob('inflammation*.csv'))
```

    ['inflammation-05.csv', 'inflammation-12.csv', 'inflammation-04.csv', 'inflammation-08.csv', 'inflammation-10.csv', 'inflammation-06.csv', 'inflammation-09.csv', 'inflammation-01.csv', 'inflammation-07.csv', 'inflammation-11.csv', 'inflammation-03.csv', 'inflammation-02.csv']



```python
# This part is not working correctly (No error messages)
# This makes figures similar to what we made before
# But this time we are making graphs for multiple data sets at once using GLOB

import glob
import numpy
import matplotlib.pyplot

filenames = sorted(glob.glob('inflamation*.csv'))
filenames = filenames[0:3]

for filename in filenames:
    print(filename)
    
    data = numpy.loadtxt(fname=filename, delimiter = ',')
    
    fig = matplotlib.pyplot.figure(figsize = (10.0, 3.0))
    
    axes1 = fig.add_subplot(1, 3, 1)
    axes2 = fig.add_subplot(1, 3, 2)
    axes3 = fig.add_subplot(1, 3, 3)
    
    axes1.set_ylabel('average')
    axes1.plot(numpy.mean(data, axis = 0))
    
    axes2.set_ylabel('max')
    axes2.plot(numpy.amax(data, axis = 0))
    
    axes3.set_ylabel('min')
    axes3.plot(numpy.amin(data, axis = 0))
    
    fig.tight_layout()
    matplotlib.pyplot.show()
    

```


```python

```



## Storing Values in Lists

```python
# This lesson we will explore lists
# Make a list by:

odds = [1, 3, 5, 7]
print('odds are:', odds)
```

    odds are: [1, 3, 5, 7]



```python
print('first element:', odds[0])
print('last element:', odds[3])
print('"-1" element:', odds[-3])
# Going negative wraps around
```

    first element: 1
    last element: 7
    "-1" element: 3



```python
# Lists can store other variable types also, like strings

names = ['Curie', 'Dawrin', 'Turing']

print('original names:', names)

names[1] = 'Jon'  # Elements can be changed

print('final names:', names)
```

    original names: ['Curie', 'Dawrin', 'Turing']
    final names: ['Curie', 'Jon', 'Turing']



```python
#name = 'Darwin'
#name[0] = 'd' 

# This part doesn't work
```


```python
# Adding values to the end of a list

odds.append(11)
print('odds after adding a value:', odds)
```

    odds after adding a value: [1, 3, 5, 7, 11]



```python
# Specific elements can be removed from lists

removed_element = odds.pop(0)
print('odds after removing the first element:', odds)
print('removed_element:', removed_element)
```

    odds after removing the first element: [3, 5, 7, 11]
    removed_element: 1



```python
# Reversing the order of elements in a list

odds.reverse()
print('odds after reverseing:', odds)
```

    odds after reverseing: [11, 7, 5, 3]



```python
odds = [3, 5, 7]
primes = odds
primes.append(2)
print('primes:', primes)
print('odds:', odds)
```

    primes: [3, 5, 7, 2]
    odds: [3, 5, 7, 2]



```python
odds = [3, 5, 7]
primes = list(odds)  # This version saves a copy of 'odds' in a new list
primes.append(2)
print('primes:', primes)
print('odds:', odds)
```

    primes: [3, 5, 7, 2]
    odds: [3, 5, 7]



```python
# Lets make a new list and explore it

binomial_name = "Drosophila melanogaster"
group = binomial_name[0:10]
print('group:', group)

species = binomial_name[11:23]
print('species:', species)

chromosomes = ['X', 'Y', '2', '3', '4']
autosomes = chromosomes[2:5]
print('autosomes:', autosomes)

last = chromosomes[-1]
print('last:', last)

```

    group: Drosophila
    species: melanogaster
    autosomes: ['2', '3', '4']
    last: 4



```python
# Looking at slices of lists

date = 'Monday 4 January 2023'
day = date[0:6]
print('Using 0 to begin range:', day)
day = date[:6]
print('Omitting beginning index:', day)
```

    Using 0 to begin range: Monday
    Omitting beginning index: Monday



```python
# More slices

months = ['jan', 'feb', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
sond = months[8:12]
print('With known last position:', sond)

sond = months[8:len(months)]
print('using len() to get last entry:', sond)
sond = months[8:]
print('Omitting ending index:', sond)
```

    With known last position: ['oct', 'nov', 'dec']
    using len() to get last entry: ['oct', 'nov', 'dec']
    Omitting ending index: ['oct', 'nov', 'dec']



```python

```




## Functions Pt 2-4

```python
# Don't forget to import packages
# Lets make a function that plots data from the first three files names 'inflammation' in this folder

import numpy
import glob
import matplotlib
import matplotlib.pyplot
```


```python
def visualise(filename):
    
    data = numpy.loadtxt(fname = filename, delimiter = ',')
    
    fig = matplotlib.pyplot.figure(figsize=(10.0, 3.0))
    
    axes1 = fig.add_subplot(1, 3, 1)
    axes2 = fig.add_subplot(1, 3, 2)
    axes3 = fig.add_subplot(1, 3, 3)
    
    axes1.set_ylabel('average')
    axes1.plot(numpy.mean(data, axis = 0))
    
    axes2.set_ylabel('max')
    axes2.plot(numpy.amax(data, axis = 0))
    
    axes3.set_ylabel('min')
    axes3.plot(numpy.min(data, axis = 0))
    
    fig.tight_layout()
    matplotlib.pyplot.show()
```


```python
def detect_problems(filename):
    
    data = numpy.loadtxt(fname = filename, delimiter = ',')
    
    if numpy.amax(data, axis = 0)[0] == 0 and numpy.amax(data, axis = 0)[20] ==20:
        print('Suspicious looking maxima!')
    elif numpy.sum(numpy.amin(data, axis = 0)) == 0:
        print('Minima add up to zero!')
    else:
        print('Seems OK')
```


```python
filenames = sorted(glob.glob('inflammation*.csv'))

for filename in filenames[:3]:
    print(filename)
    visualise(filename)
    detect_problems(filename)
```

    inflammation-01.csv



![png](output_3_1.png)


    Suspicious looking maxima!
    inflammation-02.csv



![png](output_3_3.png)


    Suspicious looking maxima!
    inflammation-03.csv



![png](output_3_5.png)


    Minima add up to zero!



```python
def offset_mean(data, target_mean_value):
    return (data - numpy.mean(data)) + target_mean_value
```


```python
z = numpy.zeros((2, 2))
print(offset_mean(z, 3))
```

    [[3. 3.]
     [3. 3.]]



```python
data = numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')
print(offset_mean(data, 0))
```

    [[-6.14875 -6.14875 -5.14875 ... -3.14875 -6.14875 -6.14875]
     [-6.14875 -5.14875 -4.14875 ... -5.14875 -6.14875 -5.14875]
     [-6.14875 -5.14875 -5.14875 ... -4.14875 -5.14875 -5.14875]
     ...
     [-6.14875 -5.14875 -5.14875 ... -5.14875 -5.14875 -5.14875]
     [-6.14875 -6.14875 -6.14875 ... -6.14875 -4.14875 -6.14875]
     [-6.14875 -6.14875 -5.14875 ... -5.14875 -5.14875 -6.14875]]


print('oringinal min, mean, and max are:', 
    numpy.amin(data), 
    numpy.mean(data), 
    numpy.mean(data), 
    numpy.amax(data))
    
offset_data = offset_mean(data, 0)

print('min, mean, and max of offset data are:',
    numpy.amin(offset_data),
    numpy.mean(offset_data),
    numpy.amax(offset_data))


```python
print('oringinal min, mean, and max are:', numpy.amin(data), numpy.mean(data), numpy.mean(data), numpy.amax(data))

offset_data = offset_mean(data, 0)

print('min, mean, and max of offset data are:', numpy.amin(offset_data), numpy.mean(offset_data), numpy.amax(offset_data))
```

    oringinal min, mean, and max are: 0.0 6.14875 6.14875 20.0
    min, mean, and max of offset data are: -6.14875 2.842170943040401e-16 13.85125



```python
print('difference in standard deviation before and after:',
      numpy.std(data) - numpy.std(offset_data))
```

    difference in standard deviation before and after: 0.0



```python
# Offset_mean(data, target_mean_value):
# return a new array containing the orininal data with its mean offset to match the desired value

def offset_mean(data, target_mean_value):
    return(data - numpy.mean(data)) + targer_mean_value
```


```python
def offset_mean(data, targer_mean_value):
    '''Return a new array containing the original data with its mean offset to match the desired value'''  # This kind of note can be written for use in help messages
    return(data - numpy.mean(data)) + targer_mean_value
```


```python
help(offset_mean)
```

    Help on function offset_mean in module __main__:
    
    offset_mean(data, targer_mean_value)
        Return a new array containing the original data with its mean offset to match the desired value
    



```python
def offset_mean(data, target_mean_value):
    '''Return a new array containing the original data 
    with its mean offset to match the desired value.
    
    Examples
    ---------
    
    >>> Offset_mean([1, 2, 3]), 0)
    array([-1., 0., 1.])
    '''''
    
    return(data - numpy.mean(data)) + target_mean_value
```


```python
help(offset_mean)
```

    Help on function offset_mean in module __main__:
    
    offset_mean(data, target_mean_value)
        Return a new array containing the original data 
        with its mean offset to match the desired value.
        
        Examples
        ---------
        
        >>> Offset_mean([1, 2, 3]), 0)
        array([-1., 0., 1.])
    



```python
numpy.loadtxt('inflammation-01.csv', delimiter = ',')
```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python
# New parameter added here --------------\

def offset_mean(data, target_mean_value = 0.0):
    '''Return a new array containing the original data 
    with its mean offset to match the desired value, (0 by default.
    
    Examples
    ---------
    
    >>> Offset_mean([1, 2, 3]), 0)
    array([-1., 0., 1.])
    '''''
    
    return(data - numpy.mean(data)) + target_mean_value
```


```python
test_data = numpy.zeros((2, 2))
print(offset_mean(test_data, 3))
```

    [[3. 3.]
     [3. 3.]]



```python
print(offset_mean(test_data))
```

    [[0. 0.]
     [0. 0.]]



```python
def display(a=1, b=2, c=3):
    print('a:', a, 'b:', b, 'c:', c)
    
print('no parameters:')
display()
print('one parameter:')
display(55)
print('two parameters:')
display(55, 66)
```

    no parameters:
    a: 1 b: 2 c: 3
    one parameter:
    a: 55 b: 2 c: 3
    two parameters:
    a: 55 b: 66 c: 3



```python
print('only setting the value of c')
display(c = 77)
```

    only setting the value of c
    a: 1 b: 2 c: 77



```python
numpy.loadtxt('inflammation-01.csv', delimiter = ',')
```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python

```


## Analyzing Patient Data Pt 1

```python
import numpy
```


```python
numpy.loadtxt(fp = open('inflammation-01.csv'), delimiter = ',')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-66cd3de96d3f> in <module>
    ----> 1 numpy.loadtxt(fp = open('inflammation-01.csv'), delimiter = ',')
    

    NameError: name 'numpy' is not defined



```python

```



## Analyzing Patient Data Pts 2-3

This is the Analyzing Patient Data lesson

```python
# Start by preping relevant packages

import numpy
```


```python
# Load the data file
# This only works when python is in the correct directory

numpy.loadtxt(fname = 'inflammation-01.csv', delimiter=',')
```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python
# Set our data variable as the file data

data = numpy.loadtxt(fname = 'inflammation-01.csv', delimiter=',')
```


```python
print(data)
```

    [[0. 0. 1. ... 3. 0. 0.]
     [0. 1. 2. ... 1. 0. 1.]
     [0. 1. 1. ... 2. 1. 1.]
     ...
     [0. 1. 1. ... 1. 1. 1.]
     [0. 0. 0. ... 0. 2. 0.]
     [0. 0. 1. ... 1. 1. 0.]]



```python
# Note that the data in "data" is an array

print(type(data))
```

    <class 'numpy.ndarray'>



```python
# Checks the bounds of the array (starts at 0,0)

print(data.shape)
```

    (60, 40)



```python
# Look at values in specific locations of the array with [ ]

print('first value in data:', data[0,0])
```

    first value in data: 0.0



```python
print('middle value in data:', data[29,19])
```

    middle value in data: 16.0



```python
# Look at a slice of the array values with [xStart:x-1, x:x]
print(data[0:4, 0:10])
```

    [[0. 0. 1. 3. 1. 2. 4. 7. 8. 3.]
     [0. 1. 2. 1. 2. 1. 3. 2. 2. 6.]
     [0. 1. 1. 3. 3. 2. 6. 2. 5. 9.]
     [0. 0. 2. 0. 4. 2. 2. 1. 6. 7.]]



```python
print(data[5:10, 0:10])
```

    [[0. 0. 1. 2. 2. 4. 2. 1. 6. 4.]
     [0. 0. 2. 2. 4. 2. 2. 5. 5. 8.]
     [0. 0. 1. 2. 3. 1. 2. 3. 5. 3.]
     [0. 0. 0. 3. 1. 5. 6. 5. 5. 8.]
     [0. 1. 1. 2. 1. 3. 5. 3. 5. 8.]]



```python
small = data[:3, 36:]
```


```python
print('small is:')
```

    small is:



```python
print(small)
```

    [[2. 3. 0. 0.]
     [1. 1. 0. 1.]
     [2. 2. 1. 1.]]



```python
# Lets use a numpy function
print(numpy.mean(data))
```

    6.14875



```python
# You can mash variable functions together separated by commas
# Equivalent to:
# maxval = numpy.amax(data)
# minval = numpy.amin... etc

maxval, minval, stdval = numpy.amax(data), numpy.amin(data), numpy.std(data)
```


```python
print(maxval, ':', minval, ':', stdval)
```

    20.0 : 0.0 : 4.613833197118566



```python
print('maximum inflammation:', maxval)
print('minimim inflammation:', maxval)
print('standard deviation:', stdval)
```

    maximum inflammation: 20.0
    minimim inflammation: 20.0
    standard deviation: 4.613833197118566



```python
# Sometimes we want to look for variation in statisticl values, such as maximum inflammation per patient, or average from day one

patient_0 = data[0, :] # 0 on the first axis (rows), everything on the second (columns)

print('maximum inflamation for patient 0:', numpy.amax(patient_0))
```

    maximum inflamation for patient 0: 18.0



```python
print('maximum inflamation for patient 2:', numpy.amax(data[2, :]))
```

    maximum inflamation for patient 2: 19.0



```python
# functions can beapplied across rows like this

print(numpy.mean(data, axis = 0))
```

    [ 0.          0.45        1.11666667  1.75        2.43333333  3.15
      3.8         3.88333333  5.23333333  5.51666667  5.95        5.9
      8.35        7.73333333  8.36666667  9.5         9.58333333 10.63333333
     11.56666667 12.35       13.25       11.96666667 11.03333333 10.16666667
     10.          8.66666667  9.15        7.25        7.33333333  6.58333333
      6.06666667  5.95        5.11666667  3.6         3.3         3.56666667
      2.48333333  1.5         1.13333333  0.56666667]



```python
# average imflammation per day across all patients

print(numpy.mean(data, axis = 0).shape)
```

    (40,)



```python
# average per person

print(numpy.mean(data, axis = 1))
```

    [5.45  5.425 6.1   5.9   5.55  6.225 5.975 6.65  6.625 6.525 6.775 5.8
     6.225 5.75  5.225 6.3   6.55  5.7   5.85  6.55  5.775 5.825 6.175 6.1
     5.8   6.425 6.05  6.025 6.175 6.55  6.175 6.35  6.725 6.125 7.075 5.725
     5.925 6.15  6.075 5.75  5.975 5.725 6.3   5.9   6.75  5.925 7.225 6.15
     5.95  6.275 5.7   6.1   6.825 5.975 6.725 5.7   6.25  6.4   7.05  5.9  ]



```python

```
