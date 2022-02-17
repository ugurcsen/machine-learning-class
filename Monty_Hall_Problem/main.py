import random

# Monty hall problem test

# Choose first selection
win = 0
for i in range(10000):
    gates = [0, 0, 0]
    gates[random.randint(0, 2)] = 1
    choose = random.randint(0, 2)

    if gates[choose] == 1:
        win += 1

# Win rate
print(win * 100 / 10000)

# Choose second select
win = 0
for i in range(10000):
    gates = [0, 0, 0]
    gates[random.randint(0, 2)] = 1
    fChoose = random.randint(0, 2)
    secChoose = 0;
    while True:
        sChoose = random.randint(0, 2)
        if sChoose != fChoose and gates[sChoose] == 0:
            break

    for x in range(3):
        if x != sChoose and x != fChoose:
            secChoose = x

    if gates[secChoose] == 1:
        win += 1

# Win rate
print(win * 100 / 10000)
