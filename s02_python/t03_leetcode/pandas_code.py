import pandas as pd

data = [['0', 'Y', 'N'], ['1', 'Y', 'Y'], ['2', 'N', 'Y'], ['3', 'Y', 'Y'], ['4', 'N', 'N']]
products = pd.DataFrame(data, columns=['product_id', 'low_fats', 'recyclable']).astype({'product_id':'int64', 'low_fats':'category', 'recyclable':'category'})

print(products)

lf = products['low_fats'] == 'Y'
r = products['recyclable'] == 'Y'
products.loc[lf & r, 'product_id']