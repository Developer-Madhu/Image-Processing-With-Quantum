gray_img = [[0,3],[5,2]]
res = []

def displayElems(mat1, mat2):
    for rows in mat1:
        tempRows = []
        for elems in rows:
            tempRows.append(f"{elems:08b}")
        mat2.append(tempRows)

displayElems(gray_img, res)
print(res)
