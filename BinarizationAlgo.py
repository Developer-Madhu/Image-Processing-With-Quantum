grey_img = [[0,3],[5,2]]
res_img = []

def processElems(mat1, mat2):
    for rows in mat1:
        tempRows = []
        for elems in rows:
            tempRows.append(f"{elems:08b}")
        mat2.append(tempRows)

def displayValuesOfMatrix(mat):
    for i in mat:
        for j in i:
            print(j)

def binarization():
    ...            

processElems(grey_img, res_img)
displayValuesOfMatrix(res_img)