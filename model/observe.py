
import numpy as np
import pandas as pd



df1 = pd.DataFrame([['A','V',3,8],['A','V',6,7],['A','W',9,6],['B','L',1,3],['B','M',5,4],['B','L',7,4]],
                                                                        columns = ['D','B','C','E'],
                                                                        index = ['one','two','three','four','five','six'])
print df1
# df2 = df1[['C','E']].groupby([df1['D'],df1['B']]).agg(['mean','sum'])
# print df2
# df2 = df2.unstack()
# print df2
print df1
df2 = df1[['D','C','E']].groupby(df1['D']).agg(['mean','sum'])
print df2