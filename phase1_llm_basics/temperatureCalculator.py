import random
from math import e


Temperature = float(input("Enter The Temperature: "))
Logit = [8,6,2]
# for _ in range(3):
#     Logit.append(random.randint(1,10))

Sum = 0
for token_score in Logit:
    Sum += e ** (token_score / Temperature)

    

for i, token in enumerate(Logit, start=1):
    probability = (e ** (token / Temperature)) / Sum
    percentage = probability * 100
 
    print(f"Token_{i}: %{percentage:.2f}")
    