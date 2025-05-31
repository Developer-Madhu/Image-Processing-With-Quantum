# # # number = 100
# # # print(format(number , 'b'))
# # matrix = [[1, 2], [3, 4]]
# # bin_matrix = []

# # def displayMatrixElems(matrix, bin_matrix):
# #     for rows in matrix:
# #         bin_rows = []
# #         for elems in rows:
# #             bin_rows.append(f"{elems:08b}")
# #         bin_matrix.append(bin_rows)

# #     # for row in bin_matrix:
# #     #     for value in row:
# #     #         print(value)


# # # def AddMatrixElems(bin_matrix):
# # #     first_row = bin_matrix[0]
# # #     second_row = bin_matrix[1]
# # #     for a in first_row:
# # #         for b in first_row:
# # #             temp = a+b
# # #         return temp
# # #     for m in second_row:
# # #         for n in second_row:
# # #             val = m+n
# # #         return val
        
# # #     return temp+val

# # def AddMatrixElems(bin_matrix):
# #     result = []
# #     for row in bin_matrix:
# #         row_sum = 0
# #         for elem in row:
# #             row_sum += int(elem, 2)  # Convert binary string to int and add
# #         result.append(row_sum)
# #     return result

# # displayMatrixElems(matrix, bin_matrix)
# # addval = AddMatrixElems(bin_matrix)
# # print(f"{addval}")

# # print(bin_matrix)

# matrix = [[1, 2], [3, 4]]
# bin_matrix = []

# # Function to convert matrix elements to 8-bit binary
# def displayMatrixElems(matrix, bin_matrix):
#     for row in matrix:
#         bin_row = []
#         for elem in row:
#             bin_row.append(f"{elem:08b}")
#         bin_matrix.append(bin_row)

# # Function to add all elements in the binary matrix
# def AddMatrixElems(bin_matrix):
#     total = 0
#     count = 0
#     for row in bin_matrix:
#         for elem in row:
#             total += int(elem, 2)
#             count += 1
#     print(f"Total Sum (Decimal): {total}")
#     print(f"Total Sum (Binary): {format(total, '08b')}")
#     return total, count

# # Function to calculate average, quotient, and remainder
# def DisplayAverage(total, count):
#     quotient = total // count
#     remainder = total % count

#     print(f"Average / Quotient (Decimal): {quotient}")
#     print(f"Average / Quotient (Binary): {format(quotient, '08b')}")
#     print(f"Remainder (Decimal): {remainder}")
#     print(f"Remainder (Binary): {format(remainder, '08b')}")

# # Run the program
# displayMatrixElems(matrix, bin_matrix)
# total, count = AddMatrixElems(bin_matrix)
# DisplayAverage(total, count)

# # Optional: Print binary matrix
# print("\nBinary Matrix:")
# for row in bin_matrix:
#     print(row)
matrix = [[0, 1, 2],[3, 4, 5],[6, 7, 2]]

def binaryConversion(n):
    return f'{n:08b}'

def average_to_binary(matrix):
    total = 0
    count = 0
    binary_matrix = []

    for row in matrix:
        binary_row = []
        for num in row:
            binary = binaryConversion(num)
            binary_row.append(binary)
            total += num
            count += 1
        binary_matrix.append(binary_row)

    average = total // count
    average_binary = binaryConversion(average)

    return binary_matrix, average, average_binary