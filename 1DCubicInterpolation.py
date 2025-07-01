def cubic_interp_1d(p0, p1, p2, p3, t):
    a = -0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3
    b = p0 - 2.5*p1 + 2*p2 - 0.5*p3
    c = -0.5*p0 + 0.5*p2
    d = p1
    return a*t*3 + b*t*2 + c*t + d

def get_input(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a valid number.")

while True:
    print("\n--- New Interpolation ---")
    x_m1 = get_input("Enter x(-1): ")
    x0   = get_input("Enter x(0) : ")
    x1   = get_input("Enter x(1) : ")
    x2   = get_input("Enter x(2) : ")
    lam  = get_input("Enter λ between 0 and 1: ")

    if not (0 <= lam <= 1):
        print("❗ λ must be between 0 and 1. Try again.")
        continue

    result = cubic_interp_1d(x_m1, x0, x1, x2, lam)
    print(f"Interpolated value at λ = {lam} is: {result:.4f}")

    choice = input("Do you want to continue? (y/n): ").strip().lower()
    if choice != 'y':
        print("PROGRAM TERMINATED BY USER!")
        break